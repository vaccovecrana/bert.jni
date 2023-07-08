plugins { id("io.vacco.oss.gitflow") version "0.9.8" }

apply(plugin = "io.vacco.oss.gitflow")
group = "io.vacco.bert"
version = "0.1.0-cd2170d"

configure<io.vacco.oss.gitflow.GsPluginProfileExtension> {
  addClasspathHell()
  sharedLibrary(true, false)
  addJ8Spec()
}

configure<io.vacco.cphell.ChPluginExtension> {
  resourceExclusions.add("module-info.class")
}

tasks.withType<Test> {
  minHeapSize = "512m"
  maxHeapSize = "16384m"
}
