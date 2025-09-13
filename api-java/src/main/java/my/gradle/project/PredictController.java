package my.gradle.project;

import ai.onnxruntime.*;
import org.springframework.web.bind.annotation.*;
import javax.annotation.PostConstruct;  // 添加这个导入
import javax.annotation.PreDestroy;   // 添加这个导入
import java.util.HashMap;
import java.util.Map;
import java.util.List;

@RestController
@RequestMapping("/predict_behavior")
public class PredictController {

    private OrtSession session;
    private OrtEnvironment env;

    @PostConstruct
    public void initModel() {
        try {
            String modelPath = "/root/Bitus-Labs/model.onnx";
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setIntraOpNumThreads(4);
            options.setInterOpNumThreads(1);
            session = env.createSession(modelPath, options);
            
            System.out.println("Model initialized successfully");
            System.out.println("Input names: " + session.getInputNames());
        } catch (Exception e) {
            System.err.println("Failed to initialize model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @PreDestroy
    public void destroy() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @PostMapping
    public Map<String, Object> predict(@RequestBody Map<String, Object> inputData) {
        Map<String, Object> response = new HashMap<>();
        
        try {
          
            if (session == null) {
                throw new RuntimeException("Model not initialized properly");
            }

          
            if (!inputData.containsKey("input")) {
                throw new IllegalArgumentException("Input data must contain 'input' field");
            }

            
            Object inputObj = inputData.get("input");
            if (!(inputObj instanceof List)) {
                throw new IllegalArgumentException("Input must be a list");
            }

            List<?> inputList = (List<?>) inputObj;
            float[][][] inputDataArray = new float[1][inputList.size()][];
            
            for (int i = 0; i < inputList.size(); i++) {
                Object rowObj = inputList.get(i);
                if (!(rowObj instanceof List)) {
                    throw new IllegalArgumentException("Each input row must be a list");
                }
                
                List<?> rowList = (List<?>) rowObj;
                inputDataArray[0][i] = new float[rowList.size()];
                
                for (int j = 0; j < rowList.size(); j++) {
                    Object valueObj = rowList.get(j);
                    if (valueObj instanceof Number) {
                        inputDataArray[0][i][j] = ((Number) valueObj).floatValue();
                    } else {
                        throw new IllegalArgumentException("Input values must be numbers at position (" + i + "," + j + ")");
                    }
                }
            }

          
            System.out.println("Received input shape: [" + 
                               inputDataArray.length + ", " + 
                               inputDataArray[0].length + ", " + 
                               inputDataArray[0][0].length + "]");

            
            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputDataArray)) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("onnx::Transpose_0", inputTensor);

                try (OrtSession.Result result = session.run(inputs)) {
                    float[][] rawPrediction = (float[][]) result.get(0).getValue();
                    float[] prediction = rawPrediction[0];

                  
                    int predictedAction = -1;
                    float maxProbability = -1;
                    for (int i = 0; i < prediction.length; i++) {
                        if (prediction[i] > maxProbability) {
                            maxProbability = prediction[i];
                            predictedAction = i;
                        }
                    }

                    response.put("predicted_action", predictedAction);
                    response.put("max_probability", maxProbability);
                    response.put("status", "success");
                }
            }

        } catch (Exception e) {
          
            response.put("error", e.getMessage());
            response.put("error_type", e.getClass().getSimpleName());
            response.put("status", "failed");
            e.printStackTrace();
        }
        
        return response;
    }
}
    