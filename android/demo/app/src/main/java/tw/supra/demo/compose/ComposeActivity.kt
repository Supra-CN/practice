package tw.supra.demo.compose

import SampleData
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import tw.supra.demo.compose.conversation.Conversation
import tw.supra.demo.compose.conversation.Message
import tw.supra.demo.compose.conversation.MessageCard
import tw.supra.demo.compose.ui.theme.DemoTheme

class ComposeActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DemoTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ContentRoot(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun ContentRoot(modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Greeting(
            name = "Android",
        )
        HelloContent()
        ClickCounter(clicks = 0) {
            Log.i("supra-debug", "ClickCounter on click")
        }
        MessageCard(Message(author = "Supra", body = "developer"))
        HorizontalDivider()
        Conversation(messages = SampleData.conversationSample)
    }
}

@Composable
private fun HelloContent() {
    var name by remember { mutableStateOf("") }
    Column(modifier = Modifier.padding(16.dp)) {
        if (name.isNotEmpty()) {
            Text(
                text = "Hello $name!",
                modifier = Modifier.padding(bottom = 8.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }
        OutlinedTextField(
            value = name,
            onValueChange = { name = it },
            label = { Text("Name") }
        )

    }
}

@Composable
fun ClickCounter(clicks: Int, onClick: () -> Unit) {
    var clickCount by remember { mutableIntStateOf(clicks) }
    Button(onClick = {
        clickCount++
        Log.i("supra-debug", "ClickCounter Btb on click[$clickCount]")
        onClick()
    }) {
        Text("I've been clicked $clickCount times")
    }
}


@Composable
fun Greeting(name: String = "supra") {
    Text(
        text = "Hello $name!",
    )
}

@Preview(
    name = "Light Mode",
    showBackground = true
)
@Preview(
    name = "Dark Mode",
    uiMode = Configuration.UI_MODE_NIGHT_YES,
    showBackground = true
)

@Composable
fun ContentRootPreview() {
    DemoTheme {
        ContentRoot()
    }
}
