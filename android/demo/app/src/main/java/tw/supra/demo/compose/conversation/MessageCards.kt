package tw.supra.demo.compose.conversation

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewParameter
import androidx.compose.ui.tooling.preview.PreviewParameterProvider
import androidx.compose.ui.unit.dp
import tw.supra.demo.R

data class Message(val author: String, val body: String, val isExpanded: Boolean = false)

@Preview
@Composable
fun MessageCard(@PreviewParameter(PreviewParameterMessage::class) message: Message) {
    Row(modifier = Modifier.padding(10.dp)) {
        Image(
            painter = painterResource(id = R.drawable.avatar_16),
            contentDescription = "logo",
            modifier = Modifier
                .height(48.dp)
                .clip(CircleShape)
                .border(1.dp, MaterialTheme.colorScheme.primary, CircleShape)
        )
        Spacer(modifier = Modifier.width(10.dp))
        // We keep track if the message is expanded or not in this
        // variable
        var isExpanded by remember { mutableStateOf(message.isExpanded) }
        val surfaceColor by animateColorAsState(
            if (isExpanded) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surface,
        )

        Column(
            modifier = Modifier
                .align(Alignment.CenterVertically)
                .clickable { isExpanded = !isExpanded }
        ) {
            Text(
                text = message.author,
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.titleLarge
            )
            Surface(
                shape = MaterialTheme.shapes.medium,
                shadowElevation = 1.dp,
                // surfaceColor color will be changing gradually from primary to surface
                color = surfaceColor,
                // animateContentSize will change the Surface size gradually
                modifier = Modifier
                    .animateContentSize()
                    .padding(1.dp)
            ) {
                Text(
                    text = message.body,
                    color = if (isExpanded) MaterialTheme.colorScheme.surface else MaterialTheme.colorScheme.secondary,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(all = 4.dp),
                    // If the message is expanded, we display all its content
                    // otherwise we only display the first line
                    maxLines = if (isExpanded) Int.MAX_VALUE else 1,
                )
            }
        }
    }
}

class PreviewParameterMessage : PreviewParameterProvider<Message> {
    override val values: Sequence<Message> = sequenceOf(
        Message("supra", "abcd asdf sadfasdfas sdfsd asdfsdf asdf asdfs sdf"),
        Message("supra", "abcd asdf sadfasdfas sdfsd asdfsdf asdf asdfs sdf",true)
    )
}