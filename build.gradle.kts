plugins { id("io.vacco.oss.gitflow") version "0.9.8" }

apply(plugin = "io.vacco.oss.gitflow")
group = "io.vacco.bert"
version = "cd2170d" // sync with https://github.com/skeskinen/bert.cpp/commit/cd2170d

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addClasspathHell()
  sharedLibrary(true, false)
  addJ8Spec()
}

configure<io.vacco.cphell.ChPluginExtension> {
  resourceExclusions.add("module-info.class")
}

dependencies {
  testImplementation("io.vacco.shax:shax:1.7.30.0.0.7")
}

tasks.withType<Test> {
  minHeapSize = "512m"
  maxHeapSize = "16384m"
}
