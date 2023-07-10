import java.util.Objects;

public class BtRecord {

  public float similarity;
  public float[] embedding;
  public String text;

  public BtRecord withSimilarity(float similarity) {
    this.similarity = similarity;
    return this;
  }

  public static BtRecord from(float[] embedding, String text) {
    var r = new BtRecord();
    r.text = Objects.requireNonNull(text);
    r.embedding = embedding;
    return r;
  }

}
