package org.example

import java.io.File

object Log {
    val file = File(System.getProperty("user.home"),"/tmp/log/suCor.log").apply {
        parentFile.let {
            if (!it.isDirectory) {
                delete()
                it.mkdirs()
            }
        }
        if (!isFile) {
            delete()
            createNewFile()
        }
    }

    fun i(tag: String, log: String) =
        "$tag:: $log".let {
//            println(it)
            file.appendText("$it\n")
        }

}
