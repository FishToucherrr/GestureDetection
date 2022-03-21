package uk.co.lemberg.motiondetectionlib;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;
import java.util.concurrent.Semaphore;

public class Detector {

    public enum GestureType {
        UpDown,
        MoveRight,
        Circle,
        MoveLeft,
        ForwardBack
    }

    public interface Listener {
        void onGestureRecognized(Detector.GestureType gestureType);
    }
    public float[] prob= new float[5];

    private static final String MODEL_FILENAME = "file:///android_asset/frozen.pb";

    private static final int GESTURE_DURATION_MS = 1280000; // 1.28 sec
    private static final int GESTURE_SAMPLES = 128;

    private static final String INPUT_NODE = "x_input";
    private static final String OUTPUT_NODE = "labels_output";
    private static final String[] OUTPUT_NODES = new String[]{OUTPUT_NODE};
    private static final int NUM_CHANNELS = 3;
    private static final long[] INPUT_SIZE = {1, GESTURE_SAMPLES, NUM_CHANNELS};
    private static final String[] labels = new String[]{"Right", "Left","UpDown","ForwardBack","Circle"};

    private static final float DATA_NORMALIZATION_COEF = 9f;
    private static final int FILTER_COEF = 20;

    private final Context context;
    private final Detector.Listener listener;
    private final Handler mainHandler;

    private final float[] outputScores = new float[labels.length];
    private final float[] recordingData = new float[GESTURE_SAMPLES * NUM_CHANNELS];
    private final float[] recognData = new float[GESTURE_SAMPLES * NUM_CHANNELS];
    private final float[] filteredData = new float[GESTURE_SAMPLES * NUM_CHANNELS];
    private int dataPos = 0;

    private TensorFlowInferenceInterface inferenceInterface;
    private HandlerThread sensorHandlerThread;
    private Handler sensorHandler;

    private Thread recognitionThread;
    private final Semaphore recognSemaphore = new Semaphore(0);

    private SensorManager sensorManager;
    private Sensor accelerometer;

    private boolean recStarted;

    public Detector(Context context, Detector.Listener listener) {
        this.context = context;
        this.listener = listener;
        mainHandler = new Handler(Looper.getMainLooper());
    }

    public void start() throws Exception {
        loadTensorflow();
        getAccelerometerSensor();

        sensorHandlerThread = new HandlerThread("sensor thread");
        sensorHandlerThread.start();
        sensorHandler = new Handler(sensorHandlerThread.getLooper());

        recognitionThread = new Thread(recognitionRunnable, "recognition thread");
        recognitionThread.start();

        recStarted = sensorManager.registerListener(sensorEventListener, accelerometer,
                GESTURE_DURATION_MS / GESTURE_SAMPLES, sensorHandler);

        if (!recStarted) {
            sensorHandlerThread.quitSafely();
            recognitionThread.interrupt();

            sensorHandlerThread = null;
            recognitionThread = null;
            sensorHandler = null;

            throw new Exception("registerListener failed. Check that the device has all accelerometer, magnetometer and gyroscope sensors");
        }
    }

    public void stop() {
        if (recStarted) {
            sensorManager.unregisterListener(sensorEventListener);

            sensorHandlerThread.quitSafely();
            recognitionThread.interrupt();

            sensorHandlerThread = null;
            recognitionThread = null;
            sensorHandler = null;

            recognSemaphore.tryAcquire(); // restore counter to zero

            recStarted = false;
        }
    }

    public boolean isStarted() {
        return recStarted;
    }

    private void loadTensorflow() {
        if (inferenceInterface == null) {
            inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILENAME);
        }
    }

    private void getAccelerometerSensor() throws Exception {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        if (accelerometer == null) throw new Exception("No TYPE_LINEAR_ACCELERATION sensor found");
    }

    /**
     * called from worker thread
     */
    private final SensorEventListener sensorEventListener = new SensorEventListener() {
        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {}

        @Override
        public void onSensorChanged(SensorEvent event) {
            synchronized (recordingData) {
                recordingData[dataPos++] = event.values[0] / DATA_NORMALIZATION_COEF;
                recordingData[dataPos++] = event.values[1] / DATA_NORMALIZATION_COEF;
                recordingData[dataPos++] = event.values[2] / DATA_NORMALIZATION_COEF;
                if (dataPos >= recordingData.length) {
                    dataPos = 0;
                }
            }

            // run recognition if recognition thread is available
            if (recognSemaphore.hasQueuedThreads()) recognSemaphore.release();
        }
    };

    private final Runnable recognitionRunnable = new Runnable() {
        @Override
        public void run() {
            while (true) {
                try {
                    recognSemaphore.acquire();
                    processData();
                    recognSemaphore.release();
                }
                catch (InterruptedException ignored) {
                    break;
                }
            }
        }
    };

    /**
     * Called from worker thread
     */
    private void processData() {
        // copy recordingData to recognData arranged
        synchronized (recordingData) {
            System.arraycopy(recordingData, 0, recognData, recognData.length - dataPos, dataPos);
            System.arraycopy(recordingData, dataPos, recognData, 0, recordingData.length - dataPos);
        }

        filterData(recognData, filteredData);

        inferenceInterface.feed(INPUT_NODE, filteredData, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, outputScores);

        // there values are mutually exclusive (i.e. leftProbability + rightProbability = 1)
        float Probability[] = new float[5];

        for(int i=0;i<5;++i){
            Probability[i]=outputScores[i];
//			Probability[i]-=0.5;
//			Probability[i]*=2;
            if(Probability[i]<0) Probability[i]=0;
        }
        // convert into independent 0..1 values
//		Probability[0]=0;
//		Probability[1]=0;
        detectGestures(Probability);
    }

    private static final float RISE_THRESHOLD = 0.99f;
    private static final float FALL_THRESHOLD = 0.9f;
    private static final long MIN_GESTURE_TIME_MS = 400000; // 0.4 sec - the minimum duration of recognized positive signal to be treated as a gesture
    //private static final long GESTURES_DELAY_TIME_MS = 1000000; // 1.0 sec - minimum delay between two gestures
    private long gestureStartTime = -1;
    private Detector.GestureType gestureType = null;
    private boolean gestureRecognized = false;

    private void detectGestures(float Probability[]) {
        if (gestureStartTime == -1) {
            // not recognized yet
            if (getHighestProb(Probability) >= RISE_THRESHOLD) {
                gestureStartTime = SystemClock.elapsedRealtimeNanos();
                gestureType = getGestureType(Probability);
            }
        }
        else {
            Detector.GestureType currType = getGestureType(Probability);
            if ((currType != gestureType) || (getHighestProb(Probability) < FALL_THRESHOLD)) {
                // reset
                gestureStartTime = -1;
                gestureType = null;
                gestureRecognized = false;
            }
            else {
                // gesture continues
                if (!gestureRecognized) {
                    long gestureTimeMs = (SystemClock.elapsedRealtimeNanos() - gestureStartTime) / 1000;
                    if (gestureTimeMs > MIN_GESTURE_TIME_MS) {
                        gestureRecognized = true;
                        callListener(gestureType);
                    }
                }
            }
        }
    }

    private void callListener(final Detector.GestureType gestureType) {
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                try { listener.onGestureRecognized(gestureType); }
                catch (Throwable ignored) {}
            }
        });
    }

    private static float getHighestProb(float Probability[]) {
        float Max=0;
        for(int i=0;i<Probability.length;++i){
            Max=Math.max(Max,Probability[i]);
        }
        return Max;
    }

    private static Detector.GestureType getGestureType(float Probability[]) {
        float Max=0;
        int index=0;
        for(int i=0;i<Probability.length;++i) {
            Max=Math.max(Max,Probability[i]);
            index=(Max==Probability[i])?i:index;
        }
        return Detector.GestureType.values()[index];
    }


    private static void filterData(float input[], float output[]) {
        Arrays.fill(output, 0);

        float ir = 1.0f / FILTER_COEF;

        for (int i = 0; i < input.length; i += NUM_CHANNELS) {
            for (int j = 0; j < FILTER_COEF; j++) {
                if (i - j * NUM_CHANNELS < 0) continue;
                output[i + 0] += input[i + 0 - j * NUM_CHANNELS] * ir;
                output[i + 1] += input[i + 1 - j * NUM_CHANNELS] * ir;
                output[i + 2] += input[i + 2 - j * NUM_CHANNELS] * ir;
            }
        }
    }
}
