package io.vacco.bert;

import java.io.Closeable;
import java.io.File;
import java.util.Arrays;

/**
 * This class is NOT thread safe. Furthermore, the buffers maintained by instances of this class
 * are model dependent, and are allocated only once during the lifetime of the instance, meaning that
 * you need to copy the contents of the embeddings buffer when working with model outputs.
 */
public class BtContext implements Closeable {

  private final int nThreads;
  private final long ctxPtr;
  private final float[] embedBuffer;
  private final int[] tokenBuffer;
  private final int[] tokensRead = new int[1];

  public BtContext(File modelPath, int nThreads) {
    if (!modelPath.exists()) {
      throw new IllegalArgumentException("Model not found: " + modelPath.getAbsolutePath());
    }
    this.nThreads = nThreads;
    this.ctxPtr = Bt.bertLoadFromFile(modelPath.getAbsolutePath());
    this.tokenBuffer = new int[Bt.bertNMaxTokens(ctxPtr)];
    this.embedBuffer = new float[Bt.bertNEmbd(ctxPtr)];
  }

  public float[] eval(String sentence) {
    tokensRead[0] = 0;
    Arrays.fill(tokenBuffer, 0);
    Arrays.fill(embedBuffer, 0);
    Bt.bertTokenize(ctxPtr, sentence, tokenBuffer, tokensRead, tokenBuffer.length);
    Bt.bertEval(ctxPtr, nThreads, tokenBuffer, tokensRead[0], embedBuffer);
    return embedBuffer;
  }

  @Override public void close() {
    Bt.bertFree(ctxPtr);
  }

}
