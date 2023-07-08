import io.vacco.bert.Bt;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.util.Arrays;

import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class BtTest {

  public static final String modelPath = "/home/jjzazuet/code/bert.cpp/models/all-MiniLM-L6-v2/ggml-model-f16.bin";

  static {
    it("Loads/releases a BERT model", () -> {
      var ctx = Bt.bertLoadFromFile(modelPath);
      Bt.bertFree(ctx);
    });
    it("Encodes a sentence into an embedding", () -> {
      var ctx = Bt.bertLoadFromFile(modelPath);
      var maxTokens = Bt.bertNMaxTokens(ctx);
      var numTokens = new int[1];
      var tokens = new int[maxTokens];
      Bt.bertTokenize(ctx, "This is a prompt", tokens, numTokens, maxTokens);

      for (int i = 0; i < numTokens[0]; i++) {
        int tok = tokens[i];
        System.out.printf("%s, %n", Bt.bertVocabIdToToken(ctx, tok));
      }

      var embedding = new float[Bt.bertNEmbd(ctx)];
      Bt.bertEval(ctx, 2, tokens, numTokens[0], embedding);
      Bt.bertFree(ctx);

      System.out.println(Arrays.toString(embedding));
    });
  }
}
