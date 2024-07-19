package org.example

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.consumeEach
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random
import kotlin.system.measureTimeMillis


fun main() {
//    runBlocking {
//        val startTime = System.currentTimeMillis()
//        val job = launch(Dispatchers.Default) {
//            try {
//                repeat(1000) { i ->
//                    println("job: I'm sleeping $i ...")
//                    delay(500L)
//                }
//            } finally {
//                withContext(NonCancellable) {
//                    println("job: I'm running finally")
//                    delay(1000L)
//                    println("job: And I've just delayed for 1 sec because I'm non-cancellable")
//                }
//
//            }
//        }
//        delay(1300L) // delay a bit
//        println("main: I'm tired of waiting!")
//        job.cancelAndJoin() // cancels the job and waits for its completion
//        println("main: Now I can quit.")
//    }
//    println("Hello World!")
//    asyncAwait()

    runBlocking {
        println("Start")
        val promot = "-> "
        print(promot)
        while (readlnOrNull()?.runCmd() == true) {
            print(promot)
        }
    }
}

fun String.runCmd(): Boolean = runCatching {
    val params = split("\\s+".toRegex()).map { it.trim() }
    val cmd = runCatching { params.removeFirst() }.getOrNull()
    cmds.getOrDefault(cmd) { println("unknown cmd: $this") }(params)
    true
}.onFailure {
    println("onFailure: $it")
}.getOrDefault(false)

val cmds = mapOf<String, List<String>.() -> Unit>(
    "exit" to { throw Exception("bye") },
    "echo" to { println(this) },
    "reset" to {
        _session = null
        println("ok")
    },

    "start" to {
        session.let { session ->
            println(this)

            // 生产任务URL
            val n = getOrNull(0)?.toIntOrNull() ?: 10

            (0..n).map { "task${mainCounter.getAndIncrement()}" }.let { tasks ->
                println("send $tasks")
                session.addTaskAndStart(*tasks.toTypedArray()).also {
                    println(it)
                }
            }
        }
    },

    "add" to { session.addTask("task${mainCounter.getAndIncrement()}") },
    "pause" to { session.pause().also { println(it) } },
    "resume" to { session.resume().also { println(it) } },
    "finish" to { session.finish().also { println(it) } },
    "f" to { session.finish().also { println(it) } },
    "async" to {
        println("root start")
        CoroutineScope(Dispatchers.Default).launch {
            println("runBlocking start")
            val deferred = async {
                println("async start")
                val defferd = CompletableDeferred<String>()
                Executors.newSingleThreadExecutor().execute {
                    println("thread start: ${Thread.currentThread()}")
                    "supra".also { println("thread delay 500...") }
                    Thread.sleep(500)
                    "supra".also { println("thread delay 500 end") }
                    defferd.completeExceptionally(Exception("error in thread"))
                    defferd.complete("supra")
                    println("thread end")
                }
//                throw Exception("error in async")
                defferd.await().also { println("async await: $it") }
            }

            runCatching {
                deferred.await().also { println("runBlocking get deferred with catching: $it") }
            }.onFailure {
                println("runBlocking onFailure $it")
            }.getOrNull().also { println("runBlocking get deferred with catching: $it") }
            println("runBlocking end")
        }
        println("root end")
    },
    "sort" to {
        println("sort start")
        val pool = IntArray(10) { Random.nextInt(0, 20) }.associateWith { "v_$it" }.also { println("gen map: $it") }
            .toMutableMap()

        while (pool.isNotEmpty()) {
            pool.toSortedMap().minByOrNull { it.key }?.let {
                pool.remove(it.key)
                println("entry: $it")
            }
        }
        println("sort end")
    }
)

val mainCounter = AtomicInteger(0)

val session: Session
    get() = _session ?: Session {
        Log.i("Main", "on Session callback: =====================")
        Log.i("Main", "on Session callback: $it")
        Log.i("Main", "on Session callback: =====================")
    }.also { _session = it }

var _session: Session? = null


fun channelable() = runBlocking {
    val channel = Channel<Int>()
    launch {
        for (x in 1..5) {
//            delay(50)
            println("send $x")
            channel.send(x)
//            delay(50)
        }
        println("pre close!")
        channel.close() // we're done sending
        println("close!")
    }

    channel.consumeEach {

    }
    for (y in channel) {
        delay(1000)
        println("recv $y")
    }
    println("Done!")
}

fun children() = runBlocking {
    // launch a coroutine to process some kind of incoming request
    val request = launch {
        // it spawns two other jobs
        launch(Job()) {
            println("job1: I run in my own Job and execute independently!")
            delay(1000)
            println("job1: I am not affected by cancellation of the request")
        }
        // and the other inherits the parent context
        launch {
            delay(100)
            println("job2: I am a child of the request coroutine")
            delay(1000)
            println("job2: I will not execute this line if my parent request is cancelled")
        }
    }
    delay(500)
    request.cancel() // cancel processing of the request
    println("main: Who has survived request cancellation?")
    delay(1000) // delay the main thread for a second to see what happens
}


fun asyncAwait() = runBlocking {

    val time = measureTimeMillis {
//        val one = async { doSomethingUsefulOne() }
        val one = async(start = CoroutineStart.LAZY) { doSomethingUsefulOne() }
        val two = async(start = CoroutineStart.LAZY) { doSomethingUsefulTwo() }
        // some computation
        one.start() // start the first one
        two.start() // start the second one
        println("The answer is ${one.await() + two.await()}")
    }
    println("Completed in $time ms")
}

suspend fun doSomethingUsefulOne(): Int {
    delay(1000L) // pretend we are doing something useful here
    return 13
}

suspend fun doSomethingUsefulTwo(): Int {
    delay(1000L) // pretend we are doing something useful here, too
    return 29
}