import io.vacco.bert.Bt;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class BtTest {
  static {
    it("Loads/releases a BERT model", () -> {
      var ctx = Bt.bertLoadFromFile("/home/jjzazuet/code/bert.cpp/models/all-MiniLM-L6-v2/ggml-model-f16.bin");
      Bt.bertFree(ctx);
    });
  }
}
