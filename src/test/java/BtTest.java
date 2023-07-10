;import io.vacco.bert.BtContext;
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

  public static final File modelPath = new File("/home/jjzazuet/code/bert.cpp/models/multi-qa-MiniLM-L6-cos-v1/ggml-model-f16.bin");
  public static BtContext bt;

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
      it("Opens a BERT context", () -> {
        System.out.println("Opening BERT context");
        BtTest.bt = new BtContext(modelPath, 4);
      });
      it("Encodes a sentence into an embedding", () -> {
        var embedding = bt.eval("This is a prompt, it should get tokenized.");
        System.out.println(Arrays.toString(embedding));
        System.out.println(Arrays.toString(bt.tokenSymbols()));
      });
      it("Computes pair-wise sequence similarity", () -> {
        var sentences = new String[] {
            "Kittens are cute",
            "We want to have a cat recognition system",
            "You should use a neural network for this",
            "It's better to apply some deep learning techniques"
        };
        for (var s0 : sentences) {
          for (var s1 : sentences) {
            var vec0 = bt.evalCopy(s0);
            var vec1 = bt.evalCopy(s1);
            System.out.printf("[%.8f], %s <---> %s%n", cosineSimilarity(vec0, vec1), s0, s1);
          }
        }
      });
      it("Queries embeddings for a search term", () -> {
        var lines = Files.readAllLines(Paths.get("./src/test/resources/documents.txt"));
        var recs = lines.stream()
            .map(txt -> BtRecord.from(bt.evalCopy(txt), txt))
            .collect(Collectors.toList());
        var qText = "Should I get health insurance?";
        var query = bt.eval(qText);
        var results = recs.stream()
            .map(rec -> rec.withSimilarity(cosineSimilarity(query, rec.embedding)))
            .sorted(Comparator.comparing(rec -> -rec.similarity))
            .limit(25)
            .collect(Collectors.toList());
        System.out.printf("====> %s <====%n", qText);
        for (var rec : results) {
          System.out.printf("[%.8f] %s%n", rec.similarity, rec.text);
        }
      });
      it("Closes the BERT context", () -> {
        System.out.println("Closing BERT context");
        bt.close();
      });
    } else {
      System.out.println("Headless mode, skipping tests");
    }
  }
}
