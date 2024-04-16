package dev.vndx.bindings

private fun formatMessage(s: String): String {
    if (!s.contains("\n")) return s
    return "${s.replace("\n", "\n | ")}\n |"
}


class NativeException(private val originalMessage: String) : RuntimeException(formatMessage(originalMessage)) {
    fun getOriginalMessage(): String {
        return originalMessage
    }
}
