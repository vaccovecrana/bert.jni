#include <jni.h>
#include "bert.h"

JNIEXPORT jobject JNICALL Java_com_example_BertJNI_bertLoadFromFile(JNIEnv *env, jclass cls, jstring fname) {
  const char *filename = (*env)->GetStringUTFChars(env, fname, NULL);
  struct bert_ctx *ctx = bert_load_from_file(filename);

  (*env)->ReleaseStringUTFChars(env, fname, filename);

  jclass bertCtxClass = (*env)->FindClass(env, "com/example/BertJNI$BertCtx");
  jmethodID constructor = (*env)->GetMethodID(env, bertCtxClass, "<init>", "()V");
  jobject bertCtxObject = (*env)->NewObject(env, bertCtxClass, constructor);

  jfieldID contextPtrField = (*env)->GetFieldID(env, bertCtxClass, "contextPtr", "J");
  (*env)->SetLongField(env, bertCtxObject, contextPtrField, (jlong) ctx);

  return bertCtxObject;
}

JNIEXPORT void JNICALL Java_com_example_BertJNI_bertFree(JNIEnv *env, jobject obj, jlong contextPtr) {
  struct bert_ctx *ctx = (struct bert_ctx *) contextPtr;
  bert_free(ctx);
}

JNIEXPORT void JNICALL Java_com_example_BertJNI_bertEncode(JNIEnv *env, jclass cls, jobject bertCtx,
                                                           jint nThreads, jstring texts, jfloatArray embeddings) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  const char *text = (*env)->GetStringUTFChars(env, texts, NULL);
  jfloat *embeddingsData = (*env)->GetFloatArrayElements(env, embeddings, NULL);

  bert_encode(ctx, nThreads, text, embeddingsData);

  (*env)->ReleaseStringUTFChars(env, texts, text);
  (*env)->ReleaseFloatArrayElements(env, embeddings, embeddingsData, 0);
}

JNIEXPORT void JNICALL Java_com_example_BertJNI_bertEncodeBatch(JNIEnv *env, jclass cls, jobject bertCtx, jint nThreads,
                                                                jint nBatchSize, jint nInputs, jobjectArray texts,
                                                                jobjectArray embeddings) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  const char **cTexts = (const char **) malloc(nInputs * sizeof(const char *));
  jfloat **cEmbeddings = (jfloat **) malloc(nInputs * sizeof(jfloat *));

  for (int i = 0; i < nInputs; i++) {
    jstring text = (*env)->GetObjectArrayElement(env, texts, i);
    const char *cText = (*env)->GetStringUTFChars(env, text, NULL);

    cTexts[i] = cText;

    jfloatArray embeddingArray = (*env)->GetObjectArrayElement(env, embeddings, i);
    jfloat *cEmbeddingArray = (*env)->GetFloatArrayElements(env, embeddingArray, NULL);

    cEmbeddings[i] = cEmbeddingArray;
  }

  bert_encode_batch(ctx, nThreads, nBatchSize, nInputs, cTexts, cEmbeddings);

  for (int i = 0; i < nInputs; i++) {
    jstring text = (*env)->GetObjectArrayElement(env, texts, i);
    (*env)->ReleaseStringUTFChars(env, text, cTexts[i]);
    jfloatArray embeddingArray = (*env)->GetObjectArrayElement(env, embeddings, i);
    (*env)->ReleaseFloatArrayElements(env, embeddingArray, cEmbeddings[i], 0);
  }

  free(cTexts);
  free(cEmbeddings);
}

JNIEXPORT void JNICALL Java_com_example_BertJNI_bertTokenize(JNIEnv *env, jclass cls, jobject bertCtx, jstring text,
                                                             jintArray tokens, jintArray numTokens, jint nMaxTokens) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  const char *cText = (*env)->GetStringUTFChars(env, text, NULL);
  jint *cTokens = (*env)->GetIntArrayElements(env, tokens, NULL);
  jint *cNumTokens = (*env)->GetIntArrayElements(env, numTokens, NULL);

  bert_tokenize(ctx, cText, cTokens, cNumTokens, nMaxTokens);

  (*env)->ReleaseStringUTFChars(env, text, cText);
  (*env)->ReleaseIntArrayElements(env, tokens, cTokens, 0);
  (*env)->ReleaseIntArrayElements(env, numTokens, cNumTokens, 0);
}

JNIEXPORT void JNICALL Java_com_example_BertJNI_bertEval(JNIEnv *env, jclass cls, jobject bertCtx, jint nThreads,
                                                         jintArray tokens, jint numTokens, jfloatArray embeddings) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  jint *cTokens = (*env)->GetIntArrayElements(env, tokens, NULL);
  jfloat *cEmbeddings = (*env)->GetFloatArrayElements(env, embeddings, NULL);

  bert_eval(ctx, nThreads, cTokens, numTokens, cEmbeddings);

  (*env)->ReleaseIntArrayElements(env, tokens, cTokens, 0);
  (*env)->ReleaseFloatArrayElements(env, embeddings, cEmbeddings, 0);
}

JNIEXPORT void JNICALL Java_com_example_BertJNI_bertEvalBatch(JNIEnv *env, jclass cls, jobject bertCtx, jint nThreads,
                                                              jint nBatchSize, jobjectArray batchTokens,
                                                              jintArray numTokens, jobjectArray batchEmbeddings) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);

  jint **cBatchTokens = (jint **) malloc(nBatchSize * sizeof(jint *));
  jint *cNumTokens = (*env)->GetIntArrayElements(env, numTokens, NULL);
  jfloat **cBatchEmbeddings = (jfloat **) malloc(nBatchSize * sizeof(jfloat *));

  for (int i = 0; i < nBatchSize; i++) {
      jintArray tokenArray = (*env)->GetObjectArrayElement(env, batchTokens, i);
      jint *cTokenArray = (*env)->GetIntArrayElements(env, tokenArray, NULL);

      cBatchTokens[i] = cTokenArray;

      jfloatArray embeddingArray = (*env)->GetObjectArrayElement(env, batchEmbeddings, i);
      jfloat *cEmbeddingArray = (*env)->GetFloatArrayElements(env, embeddingArray, NULL);

      cBatchEmbeddings[i] = cEmbeddingArray;
  }

  bert_eval_batch(ctx, nThreads, nBatchSize, cBatchTokens, cNumTokens, cBatchEmbeddings);

  for (int i = 0; i < nBatchSize; i++) {
      jintArray tokenArray = (*env)->GetObjectArrayElement(env, batchTokens, i);
      (*env)->ReleaseIntArrayElements(env, tokenArray, cBatchTokens[i], 0);
      jfloatArray embeddingArray = (*env)->GetObjectArrayElement(env, batchEmbeddings, i);
      (*env)->ReleaseFloatArrayElements(env, embeddingArray, cBatchEmbeddings[i], 0);
  }

  free(cBatchTokens);
  free(cBatchEmbeddings);
  (*env)->ReleaseIntArrayElements(env, numTokens, cNumTokens, 0);
}

JNIEXPORT jint JNICALL Java_com_example_BertJNI_bertNEmbd(JNIEnv *env, jclass cls, jobject bertCtx) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  jint nEmb = bert_n_embd(ctx);
  return nEmb;
}

JNIEXPORT jint JNICALL Java_com_example_BertJNI_bertNMaxTokens(JNIEnv *env, jclass cls, jobject bertCtx) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  jint nMaxTokens = bert_n_max_tokens(ctx);
  return nMaxTokens;
}

JNIEXPORT jstring JNICALL Java_com_example_BertJNI_bertVocabIdToToken(JNIEnv *env, jclass cls, jobject bertCtx, jint id) {
  struct bert_ctx *ctx = (struct bert_ctx *) (*env)->GetLongField(env, bertCtx, contextPtrField);
  const char *token = bert_vocab_id_to_token(ctx, id);
  jstring jToken = (*env)->NewStringUTF(env, token);
  return jToken;
}
