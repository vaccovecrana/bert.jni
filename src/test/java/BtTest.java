import io.vacco.bert.Bt;
import io.vacco.bert.BtContext;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import java.awt.*;
import java.io.File;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class BtTest {

  public static final File modelPath = new File("/home/jjzazuet/code/bert.cpp/models/all-MiniLM-L6-v2/ggml-model-f16.bin");

  public static float cosineSimilarity(float[] vectorA, float[] vectorB) {
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    for (int i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += Math.pow(vectorA[i], 2);
      normB += Math.pow(vectorB[i], 2);
    }
    return dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
  }

  static {
    if (!GraphicsEnvironment.isHeadless()) {
      it("Loads/releases a BERT model", () -> {
        var ctx = Bt.bertLoadFromFile(modelPath.getAbsolutePath());
        Bt.bertFree(ctx);
      });
      it("Encodes a sentence into an embedding", () -> {
        try (var bt = new BtContext(modelPath, 2)) {
          var embedding = bt.eval("This is a prompt");
          System.out.println(Arrays.toString(embedding));
        }
      });
      it("Queries embeddings for a search term", () -> {
        try (var bt = new BtContext(modelPath, 2)) {
          var lines = Files.readAllLines(Paths.get("./src/test/resources/documents.txt"));
          var recs = lines.stream()
              .map(txt -> BtRecord.from(bt.eval(txt), txt))
              .collect(Collectors.toList());
          var qText = "Should I get health insurance?";
          var query = bt.eval(qText);
          var results = recs.stream()
              .map(rec -> rec.withSimilarity(cosineSimilarity(query, rec.embedding)))
              .sorted(Comparator.comparing(rec -> -rec.similarity))
              .limit(10)
              .collect(Collectors.toList());
          System.out.printf("====> %s <====%n", qText);
          for (var rec : results) {
            System.out.printf("[%.8f] %s%n", rec.similarity, rec.text);
          }
        }
      });
    } else {
      System.out.println("Headless mode, skipping tests");
    }
  }
}
