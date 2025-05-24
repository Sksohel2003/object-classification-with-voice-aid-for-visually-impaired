package com.btechviral.objectdetector;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Toast;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.*;

public class DetectorActivity extends MainActivity implements ImageReader.OnImageAvailableListener {

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.55f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private boolean computingDetection = false;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private long lastProcessingTimeMs;
    private long timestamp = 0;

    private Integer sensorOrientation;
    private Classifier detector;
    private MultiBoxTracker tracker;
    private BorderedText borderedText;
    private OverlayView trackingOverlay;
    private List<Pair<Pair<String, Float>, RectF>> screenRects;

    private TextToSpeech textToSpeech;
    private long lastSpeechTime = 0;
    private static final long SPEECH_TIME_GAP = 5000;

    private boolean isTextMode = false;
    private TextRecognizer textRecognizer;
    private SpeechRecognizer speechRecognizer;
    private Intent speechRecognizerIntent;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        try {
            detector = TFLiteObjectDetectionAPIModel.create(getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE, TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT).show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation, MAINTAIN_ASPECT);
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(canvas -> {
            tracker.draw(canvas);
            if (isDebug()) tracker.drawDebug(canvas);
        });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
        screenRects = tracker.getScreenRects();

        textToSpeech = new TextToSpeech(getApplicationContext(), status -> {
            if (status == TextToSpeech.SUCCESS) textToSpeech.setLanguage(Locale.US);
        });

        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

        setupVoiceCommands();
    }

    private void setupVoiceCommands() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());

        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null) {
                    for (String command : matches) {
                        if (command.equalsIgnoreCase("extract text")) {
                            isTextMode = true;
                            speak("Text extraction enabled");
                        } else if (command.equalsIgnoreCase("stop text")) {
                            isTextMode = false;
                            speak("Object detection resumed");
                        }
                    }
                }
                speechRecognizer.startListening(speechRecognizerIntent);
            }
            public void onReadyForSpeech(Bundle params) {}
            public void onBeginningOfSpeech() {}
            public void onRmsChanged(float rmsdB) {}
            public void onBufferReceived(byte[] buffer) {}
            public void onEndOfSpeech() {}
            public void onError(int error) { speechRecognizer.startListening(speechRecognizerIntent); }
            public void onPartialResults(Bundle partialResults) {}
            public void onEvent(int eventType, Bundle params) {}
        });

        speechRecognizer.startListening(speechRecognizerIntent);
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        runInBackground(() -> {
            if (isTextMode) {
                InputImage image = InputImage.fromBitmap(rgbFrameBitmap, 0);
                textRecognizer.process(image)
                        .addOnSuccessListener(visionText -> {
                            String extracted = visionText.getText();
                            long currentTime = System.currentTimeMillis();

                            // Speak only if 6 seconds have passed since last speech
                            if (!extracted.isEmpty() && (currentTime - lastSpeechTime >= 6000)) {
                                speak(extracted);
                                lastSpeechTime = currentTime;
                            }

                            computingDetection = false;
                        })
                        .addOnFailureListener(e -> computingDetection = false);
                return;
            }

            final long startTime = SystemClock.uptimeMillis();
            final List<Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas1 = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2.0f);

            final List<Recognition> mappedRecognitions = new LinkedList<>();
            List<Pair<Recognition, Float>> objectDistances = new LinkedList<>();

            for (final Recognition result : results) {
                final RectF location = result.getLocation();
                if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                    canvas1.drawRect(location, paint);
                    cropToFrameTransform.mapRect(location);
                    result.setLocation(location);
                    mappedRecognitions.add(result);

                    // Use the updated distance calculation method with object label
                    float distance = calculateDistance(location.width(), location.height(), result.getTitle());
                    objectDistances.add(new Pair<>(result, distance));
                }
            }

            Pair<Recognition, Float> nearestObject = null;
            for (Pair<Recognition, Float> pair : objectDistances) {
                if (nearestObject == null || pair.second < nearestObject.second) nearestObject = pair;
            }

            if (nearestObject != null) {
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastSpeechTime >= SPEECH_TIME_GAP) {
                    String speechText = nearestObject.first.getTitle() + " detected, at distance " + String.format(Locale.US, "%.2f", nearestObject.second) + " meters";

                    speak(speechText);
                    lastSpeechTime = currentTime;
                }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();
            computingDetection = false;
        });
    }

    private static final Map<String, Float> referenceWidths = new HashMap<>();

    static {
        referenceWidths.put("person", 0.5f); // meters
        referenceWidths.put("bicycle", 0.6f);
        referenceWidths.put("car", 1.8f);
        referenceWidths.put("motorcycle", 0.7f);
        referenceWidths.put("airplane", 35.0f); // Example, you can adjust
        referenceWidths.put("bus", 2.5f);
        referenceWidths.put("train", 2.8f);
        referenceWidths.put("truck", 2.5f);
        referenceWidths.put("boat", 5.0f);
        referenceWidths.put("traffic light", 0.3f);
        referenceWidths.put("fire hydrant", 0.25f);
        referenceWidths.put("stop sign", 0.75f);
        referenceWidths.put("parking meter", 0.4f);
        referenceWidths.put("bench", 1.2f);
        referenceWidths.put("bird", 0.3f);
        referenceWidths.put("cat", 0.3f);
        referenceWidths.put("dog", 0.6f);
        referenceWidths.put("horse", 1.4f);
        referenceWidths.put("sheep", 1.2f);
        referenceWidths.put("cow", 1.6f);
        referenceWidths.put("elephant", 4.0f);
        referenceWidths.put("bear", 1.8f);
        referenceWidths.put("zebra", 1.4f);
        referenceWidths.put("giraffe", 3.0f);
        referenceWidths.put("backpack", 0.4f);
        referenceWidths.put("umbrella", 0.5f);
        referenceWidths.put("handbag", 0.3f);
        referenceWidths.put("tie", 0.2f);
        referenceWidths.put("frisbee", 0.25f);
        referenceWidths.put("skis", 0.2f);
        referenceWidths.put("snowboard", 0.3f);
        referenceWidths.put("sports ball", 0.3f);
        referenceWidths.put("kite", 1.5f);
        referenceWidths.put("baseball bat", 0.7f);
        referenceWidths.put("baseball glove", 0.5f);
        referenceWidths.put("skateboard", 0.7f);
        referenceWidths.put("surfboard", 1.0f);
        referenceWidths.put("tennis racket", 0.2f);
        referenceWidths.put("bottle", 0.07f);
        referenceWidths.put("wine glass", 0.15f);
        referenceWidths.put("cup", 0.1f);
        referenceWidths.put("fork", 0.15f);
        referenceWidths.put("knife", 0.15f);
        referenceWidths.put("spoon", 0.15f);
        referenceWidths.put("bowl", 0.2f);
        referenceWidths.put("fruit", 0.15f); // Placeholder value for generic fruit
        referenceWidths.put("apple", 0.1f);
        referenceWidths.put("sandwich", 0.25f);
        referenceWidths.put("ball", 0.2f);
        referenceWidths.put("broccoli", 0.2f);
        referenceWidths.put("carrot", 0.2f);
        referenceWidths.put("hot dog", 0.25f);
        referenceWidths.put("pizza", 0.3f);
        referenceWidths.put("donut", 0.2f);
        referenceWidths.put("cake", 0.2f);
        referenceWidths.put("chair", 0.45f);
        referenceWidths.put("couch", 1.5f);
        referenceWidths.put("potted plant", 0.5f);
        referenceWidths.put("bed", 1.6f);
        referenceWidths.put("dining table", 1.5f);
        referenceWidths.put("table", 1.0f);
        referenceWidths.put("tv", 0.6f);
        referenceWidths.put("laptop", 0.35f);
        referenceWidths.put("mouse", 0.15f);
        referenceWidths.put("remote", 0.2f);
        referenceWidths.put("keyboard", 0.3f);
        referenceWidths.put("cell phone", 0.08f);
        referenceWidths.put("microwave", 0.6f);
        referenceWidths.put("oven", 0.6f);
        referenceWidths.put("toaster", 0.4f);
        referenceWidths.put("sink", 0.5f);
        referenceWidths.put("refrigerator", 0.7f);
        referenceWidths.put("book", 0.2f);
        referenceWidths.put("clock", 0.3f);
        referenceWidths.put("vase", 0.3f);
        referenceWidths.put("scissors", 0.2f);
        referenceWidths.put("teddy bear", 0.3f);
        referenceWidths.put("hair drier", 0.2f);
        referenceWidths.put("toothbrush", 0.1f);
        // Add more objects as needed
    }

    private float calculateDistance(float width, float height, String label) {
        float objectWidthMeters = referenceWidths.getOrDefault(label.toLowerCase(), 0.5f); // default to 0.5 meters

        // Calculate the pixel width (or height) of the object from bounding box
        float pixelSize = Math.max(width, height);  // You can also use width only

        // Camera characteristics
        float focalLengthMm = 4.15f;
        float sensorWidthMm = 6.17f; // For 1/3.2" sensor typical in phones

        // Convert focal length to pixels
        float focalLengthPixels = (previewWidth * focalLengthMm) / sensorWidthMm;

        // Avoid division by zero
        if (pixelSize == 0) return 0.0f;

        // Distance formula: (real width * focal length) / pixel width
        float distanceMeters = (objectWidthMeters * focalLengthPixels) / pixelSize;

        // Round to 2 decimal places
        return Math.round(distanceMeters * 100f) / 100f;
    }


    private void speak(String text) {
        if (textToSpeech != null) textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null);
    }

    @Override protected int getLayoutId() { return R.layout.camera_connection_fragment_tracking; }
    @Override protected Size getDesiredPreviewFrameSize() { return DESIRED_PREVIEW_SIZE; }
    @Override protected void setUseNNAPI(final boolean isChecked) { runInBackground(() -> detector.setUseNNAPI(isChecked)); }
    @Override protected void setNumThreads(final int numThreads) { runInBackground(() -> detector.setNumThreads(numThreads)); }

    @Override public synchronized void onDestroy() {
        super.onDestroy();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (speechRecognizer != null) {
            speechRecognizer.stopListening();
            speechRecognizer.destroy();
        }
    }

    private enum DetectorMode { TF_OD_API; }
}


