package org.example

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.consumeEach
import java.util.concurrent.atomic.AtomicInteger

class Session(val callback: (CallbackInfo) -> Unit) {

    val LOG_TAG = "Session"

    enum class Status { Initialized, Running, Paused, Finished }

    private var _status = Status.Initialized
        set(value) {
            Log.i(LOG_TAG, "status update :: $field => $value")
            field = value
        }

    private val scope = CoroutineScope(Dispatchers.Default)
    private val requestTaskChannel = Channel<RequestTask>(Channel.UNLIMITED)
    private val responseHandlerChannel = Channel<ResponseTask>(2)
    private val playAudioChannel = Channel<PlayTask>(1)

    private var currentPlayTask: PlayTask? = null

    private val counter = mutableMapOf<Any, AtomicInteger>()
    private val counterRunningRequestTask get() = counter.getOrCreate("COUNTER_RUNNING_REQUEST_TASK")
    private val counterRunningResponseTask get() = counter.getOrCreate("COUNTER_RUNNING_RESPONSE_TASK")
    private val counterRunningPlayTask get() = counter.getOrCreate("COUNTER_RUNNING_PLAY_TASK")

    private val allRunningCounter by lazy {
        listOf(
            counterRunningRequestTask,
            counterRunningResponseTask,
            counterRunningPlayTask
        )
    }
    private val allRunningCount get() = allRunningCounter.sumOf { it.get() }

    private fun MutableMap<Any, AtomicInteger>.count(tag: Any) = getOrPut(tag) { AtomicInteger(0) }.getAndIncrement()
    private fun MutableMap<Any, AtomicInteger>.getOrCreate(tag: Any) = getOrPut(tag) { AtomicInteger(0) }
    private val requestingCounter get() = counter.getOrCreate("requesting")
    private val playingCounter get() = counter.getOrCreate("playing")

    @Synchronized
    private fun changeStatus(status: Status, error: Throwable? = null) {
        _status = status
        callback(CallbackInfo(status, error))
    }

    @Synchronized
    fun addTaskAndStart(vararg task: String): Boolean =
        addTaskAndStart(* task.map { RequestTask(scope, it) }.toTypedArray())

    @Synchronized
    fun addTaskAndStart(vararg task: RequestTask): Boolean = addTask(* task).also { start() }

    @Synchronized
    fun addTask(vararg task: String): Boolean = addTask(* task.map { RequestTask(scope, it) }.toTypedArray())

    @Synchronized()
    fun addTask(vararg task: RequestTask) = Status.Finished != _status && task.all { task ->
        Log.i(LOG_TAG, "addTask:: ${task.id}")
        requestTaskChannel.trySend(task).isSuccess.takeIf { it }?.apply {
            counterRunningRequestTask.incrementAndGet()
        }.also { Log.i(LOG_TAG, "running[${it}] increment for itemTaskChannel") } ?: false
    }

    @Synchronized()
    fun start(): Boolean = if (Status.Initialized != _status) false else true.also { _ ->
        changeStatus(Status.Running)
        scope.launch {

            Log.i(LOG_TAG, "start [task]:: launch itemTaskChannel.consumeEach")

            requestTaskChannel.consumeEach { itemTask ->
                Log.i(LOG_TAG, "start [task${itemTask.id}]:: on consume")
                counterRunningResponseTask.incrementAndGet().also {
                    Log.i(LOG_TAG, "running[${it}] increment for responseHandlerChannel")
                }
                Log.i(LOG_TAG, "start [task${itemTask.id}]:: send deferred to responseHandlerChannel")
                runCatching {
                    responseHandlerChannel.send(ResponseTask(scope, itemTask, itemTask.fetchMediaUrlsAsync(itemTask)))
                }.onFailure { finish(it) }
                Log.i(LOG_TAG, "start [task${itemTask.id}]:: on consumed")
                counterRunningRequestTask.decrementAndGet().also {
                    Log.i(LOG_TAG, "running[${it}] Function… for counterRunningItemTask")
                }
                checkRunningOrFinish()
            }

            Log.i(LOG_TAG, "start [task]:: itemTaskChannel done")
        }

        scope.launch {
            Log.i(LOG_TAG, "start [response] :: launch responseHandlerChannel.consumeEach")

            responseHandlerChannel.consumeEach { response ->
                Log.i(LOG_TAG, "start [response${response.id}]:: on consume")
                runCatching {
                    response.result.await().forEach {
                        Log.i(LOG_TAG, "start [response${response.id}]:: send to playAudioChannel")
                        counterRunningPlayTask.incrementAndGet().also {
                            Log.i(LOG_TAG, "running[${it}] increment for playAudioChannel")
                        }
                        playAudioChannel.send(PlayTask(scope, it, response.requestTask))

                        Log.i(LOG_TAG, "start [response${response.id}]:: on consumed").also {
                            Log.i(LOG_TAG, "running[${it}] increment")
                        }
                    }
                }.onFailure { finish(it) }

                counterRunningResponseTask.decrementAndGet().also {
                    Log.i(LOG_TAG, "running[${it}] decrement for responseHandlerChannel")
                }
                checkRunningOrFinish()
            }
            Log.i(LOG_TAG, "start [response]:: launch responseHandlerChannel done")

        }

        scope.launch {
            Log.i(LOG_TAG, "start [play] :: launch playAudioChannel.consumeEach")

            playAudioChannel.consumeEach { playTask ->
                Log.i(LOG_TAG, "start [play${playTask.id}] :: on consume")

                runCatching { playTask.playMediaUrlsAsync(playTask).await() }.onFailure {
                    finish(it)
                }

                counterRunningPlayTask.decrementAndGet().also {
                    Log.i(LOG_TAG, "running[${it}] decrement for playAudioChannel")
                }
                Log.i(LOG_TAG, "start [play${playTask.id}] :: on consumed")

                checkRunningOrFinish()
            }

            Log.i(LOG_TAG, "start [play] :: playAudioChannel done")
        }
    }

    @Synchronized()
    fun pause() = (Status.Running == _status) && currentPlayTask?.pause()?.takeIf { it }.also {
        changeStatus(Status.Paused)
    } ?: false

    @Synchronized()
    fun resume() = (Status.Paused == _status) && currentPlayTask?.resume()?.takeIf { it }.also {
        changeStatus(Status.Running)
    } ?: false


    @Synchronized()
    fun finish(error: Throwable? = null) = if (Status.Finished == _status) false else true.also {
        changeStatus(Status.Finished, error)
        counter.clear()
        requestTaskChannel.cancel()
        playAudioChannel.cancel()
        scope.cancel()
    }

    @Synchronized
    private fun checkRunningOrFinish() = (allRunningCount.also { Log.i(LOG_TAG, "all running count[${it}]") } < 1) &&
            finish().also { Log.i(LOG_TAG, "finish by no more task") }


    inner class ResponseTask(
        val scope: CoroutineScope,
        val requestTask: RequestTask,
        val result: Deferred<List<String>>
    ) {
        val id = "${requestTask.id}-${counter.count(this::class)}"
    }

    inner class RequestTask(val scope: CoroutineScope, val text: String) {
        val id = counter.count(this::class)

        // 模拟的异步网络请求函数
        private fun request(requestTask: RequestTask, callback: (List<String>) -> Unit) {
            scope.launch {
                val totalInMs = (500L..5000L).random()
                var delayInMs = totalInMs
                val waitStep = 500L

                val log = { id: Any, tag: String ->
                    Log.i(LOG_TAG, "request[${id} / ${requestingCounter}] => remaining: $delayInMs/${totalInMs} $tag")
                }
                Log.i(LOG_TAG, "request => remaining: $delayInMs ")
                log(requestTask.id, "...")

                while (delayInMs > waitStep) {
                    log(requestTask.id, "...")
                    delay(waitStep)
                    delayInMs -= waitStep
                }
                delay(delayInMs)
                log(requestTask.id, "done!")

                val taskUrl = requestTask.text
                callback(listOf("$taskUrl/media1", "$taskUrl/media2", "$taskUrl/media3")) // 返回媒体URL列表
            }
        }


        // 异步获取任务URL对应的媒体URL列表
        suspend fun fetchMediaUrlsAsync(requestTask: RequestTask): Deferred<List<String>> = scope.async {
            val completableDeferred = CompletableDeferred<List<String>>()
            Log.i(LOG_TAG, "request increment: ${requestingCounter.incrementAndGet()}")
            request(requestTask) { mediaUrls ->
                completableDeferred.complete(mediaUrls)
                Log.i(LOG_TAG, "request decrement: ${requestingCounter.decrementAndGet()}")
            }
            completableDeferred.await() // 等待并返回结果
        }
    }


    inner class PlayTask(val scope: CoroutineScope, val mediaUrl: String, val requestTask: RequestTask) {
        val id = "${requestTask.id}-${counter.count(this::class)}"

        var pausing = false

        fun pause() = true.also {
            Log.i(LOG_TAG, "play[${id}] => pause")
            pausing = true
        }

        fun resume() = true.also {
            Log.i(LOG_TAG, "play[${id}] => pause")
            pausing = false
        }

        // 模拟的媒体播放函数
        fun play(playTask: PlayTask, finishCallback: (String) -> Unit) {
            scope.launch {
                val totalInMs = (500L..10000L).random()
                var delayInMs = totalInMs
                val waitStep = 500L
                val log = { id: Any, tag: String ->
                    Log.i(LOG_TAG, "play[${id} / ${playingCounter}] => remaining: $delayInMs/${totalInMs} $tag")
                }
                Log.i(LOG_TAG, "play[${id} / ${playingCounter}] => remaining: $delayInMs/${totalInMs}")
                while (delayInMs > waitStep) {
                    log(playTask.id, if (pausing) "pausing..." else "...")
                    delay(waitStep)
                    if (!pausing) {
                        delayInMs -= waitStep
                    }
                }

//                throw error("by supra")

                delay(delayInMs)
                log(playTask.id, "done!")

                finishCallback(playTask.mediaUrl) // 通知播放完成
            }
        }

        // 异步获取任务URL对应的媒体URL列表
        suspend fun playMediaUrlsAsync(playTask: PlayTask) = scope.async {
            val completableDeferred = CompletableDeferred<String>()
            Log.i(LOG_TAG, "play increment: ${playingCounter.incrementAndGet()}")
            currentPlayTask = playTask
            Log.i(LOG_TAG, "play currentPlayTask: ${playTask.id}")

            play(playTask) { mediaUrls ->
                Log.i(LOG_TAG, "play currentPlayTask release: ${currentPlayTask?.id}")
                currentPlayTask = null
                completableDeferred.complete(mediaUrls)
                Log.i(LOG_TAG, "play decrement: ${playingCounter.decrementAndGet()}")
            }
            completableDeferred.await() // 等待并返回结果
        }

    }

    data class CallbackInfo(val status: Status, val error: Throwable? = null) {
        override fun toString(): String {
            val errorInfo = error?.let { Exception(error).stackTraceToString() }
            return "${super.toString()} error $errorInfo\n"
        }
    }

}

