package tw.supra.practice

import java.util.concurrent.atomic.AtomicInteger

sealed class Item<DataType>(val typeId: Int, val data: DataType) {
 class DramaItem(dramaData: String) :
  Item<String>(typeCounter.getAndIncrement(), dramaData)

 class OptItem(drawableRes: Int) :
  Item<Int>(typeCounter.getAndIncrement(), drawableRes)

 companion object {
  private var typeCounter = AtomicInteger(0)
 }
}