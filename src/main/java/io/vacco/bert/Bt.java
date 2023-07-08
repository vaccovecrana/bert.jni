package io.vacco.bert;

public class Bt {
  static {
    System.loadLibrary("bert"); // Replace "bert" with the actual library name
  }

  public static class BertCtx {
    private long contextPtr;
    private native void bertFree(long contextPtr);
    public void free() {
      bertFree(contextPtr);
    }
  }

  public static native BertCtx bertLoadFromFile(String fname);

  public static native void bertEncode(BertCtx ctx, int nThreads, String texts, float[] embeddings);

  public static native void bertEncodeBatch(BertCtx ctx, int nThreads, int batchSize, int inputsSize, String[] texts, float[][] embeddings);

  public static native void bertTokenize(BertCtx ctx, String text, int[] tokens, int[] numTokens, int maxTokens);

  public static native void bertEval(BertCtx ctx, int nThreads, int[] tokens, int numTokens, float[] embeddings);

  public static native void bertEvalBatch(BertCtx ctx, int nThreads, int batchSize, int[][] batchTokens, int[] numTokens, float[][] batchEmbeddings);

  public static native int bertNEmbd(BertCtx ctx);

  public static native int bertNMaxTokens(BertCtx ctx);

  public static native String bertVocabIdToToken(BertCtx ctx, int id);
}
