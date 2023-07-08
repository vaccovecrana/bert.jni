package io.vacco.bert;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;

public class Bt {

  public static final String NATIVE_FOLDER_PATH_PREFIX = "bert-jni";
  private static File temporaryDir;

  public static void loadLibsFromJar(String ... paths) {
    try {
      if (temporaryDir == null) {
        temporaryDir = new File(System.getProperty("java.io.tmpdir"), NATIVE_FOLDER_PATH_PREFIX);
        temporaryDir.deleteOnExit();
      }
      for (var path : paths) {
        var parts = path.split("/");
        var fName = (parts.length > 1) ? parts[parts.length - 1] : null;
        var temp = new File(temporaryDir, fName);
        temp.mkdirs();
        try (var is = Bt.class.getResourceAsStream(path)) {
          Files.copy(is, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
          System.load(temp.getAbsolutePath());
        }
      }
    } catch (Exception e) {
      throw new IllegalStateException("Unable to load native dependencies: " + Arrays.toString(paths), e);
    }
  }

  static {
    var os = String.format("%s-%s", System.getProperty("os.name"), System.getProperty("os.arch"));
    switch (os) {
      case "Linux-amd64":
        loadLibsFromJar("/io/vacco/bert/libbert-jni.so");
        break;
      default:
        var msg = String.format("No native binaries available for [%s]. PRs are welcome :)", os);
        System.err.println(msg);
    }
  }

  public static native void bertFree(long contextPtr);

  public static native long bertLoadFromFile(String fname);

  public static native void bertEncode(long ctx, int nThreads, String texts, float[] embeddings);

  public static native void bertEncodeBatch(long ctx, int nThreads, int batchSize, int inputsSize, String[] texts, float[][] embeddings);

  public static native void bertTokenize(long ctx, String text, int[] tokens, int[] numTokens, int maxTokens);

  public static native void bertEval(long ctx, int nThreads, int[] tokens, int numTokens, float[] embeddings);

  public static native void bertEvalBatch(long ctx, int nThreads, int batchSize, int[][] batchTokens, int[] numTokens, float[][] batchEmbeddings);

  public static native int bertNEmbd(long ctx);

  public static native int bertNMaxTokens(long ctx);

  public static native String bertVocabIdToToken(long ctx, int id);

}
