# bert.jni

Provides JNI bindings for [bert.cpp](https://github.com/skeskinen/bert.cpp).

Available in [Maven Central](https://mvnrepository.com/artifact/io.vacco.bert).

```
implementation("io.vacco.bert:bert.jni:<LATEST_VERSION>>")
```

Since `bert.cpp` does not appear to be tracking release versions, this library will be in sync with specific commits from its repository.

    0.1.0-cd2170d

Where `0.1.0` is the version number for this library, and `cd2170d` is the `bert.cpp` commit used to build the native code.

See [test cases](./src/test/java/BtTest.java) for example use.

The following `bert.cpp` models appear to work correctly when loading their `f16` variants.

- `all-MiniLM-L12-v2`
- `all-MiniLM-L6-v2`
- `multi-qa-MiniLM-L6-cos-v1`

## Caveats

- Only Linux is supported at the moment. PRs for other Operating systems are welcome and encouraged.
- A custom build for `ggml`, `bert.cpp` and the JNI bindings is needed until [this issue](https://github.com/skeskinen/bert.cpp/issues/17) gets resolved.
- Not all converted GGML models may load successfully. Please open an issue if a specific model crashes.

- Initial release.
- Built with `bert.cpp` at commit [cd2170](https://github.com/skeskinen/bert.cpp/commit/cd2170).