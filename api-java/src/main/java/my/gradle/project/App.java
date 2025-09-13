package my.gradle.project;

import ai.onnxruntime.*;
import java.util.Map;
import java.util.HashMap;

public class App {

    public String getGreeting() {
        return "Hello, Gradle!";
    }

    public static void main(String[] args) {
        try {
        
            String modelPath = "/root/Bitus-Labs/model.onnx";

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      
            options.setIntraOpNumThreads(4);
            options.setInterOpNumThreads(1);
            
            OrtSession session = env.createSession(modelPath, options);

            System.out.println("Model inputs:");
            for (NodeInfo inputInfo : session.getInputInfo().values()) {
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                System.out.println("Input name: " + inputInfo.getName());
                System.out.println("Input shape: " + java.util.Arrays.toString(tensorInfo.getShape()));
            }

          
            System.out.println("Model outputs:");
            for (NodeInfo outputInfo : session.getOutputInfo().values()) {
                TensorInfo tensorInfo = (TensorInfo) outputInfo.getInfo();
                System.out.println("Output name: " + outputInfo.getName());
                System.out.println("Output shape: " + java.util.Arrays.toString(tensorInfo.getShape()));
            }

            String inputName = "onnx::Transpose_0";

        
            float[][][] inputData = new float[][][] {
                { {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f} } 
            };

            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, inputTensor);

            OrtSession.Result result = session.run(inputs);

            float[][] rawPrediction = (float[][]) result.get(0).getValue();
          
            float[] prediction = rawPrediction[0];
            
            System.out.println("Model prediction probabilities: ");
            for (float prob : prediction) {
                System.out.printf("%.4f ", prob);
            }
            System.out.println();

          
            int predictedAction = -1;
            float maxProbability = -1;
            for (int i = 0; i < prediction.length; i++) {
                if (prediction[i] > maxProbability) {
                    maxProbability = prediction[i];
                    predictedAction = i;
                }
            }

            // 0=view, 1=cart, 2=purchase
            System.out.println("Predicted action: " + predictedAction);
            System.out.println("Maximum probability: " + maxProbability);

            inputTensor.close();
            result.close();
            session.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
    