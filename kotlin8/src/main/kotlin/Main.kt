package build.gradle.kts

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
data class Book(
    var title: String,
    var author: String?,
    var yearPublished: Int,
    var copies: Int
) {
    companion object {
        const val version: Int = 1
        var books = mutableListOf<Book>()
        fun addManually() {
            println("Enter book title:")
            val title = readLine() ?: ""
            println("Enter book author:")
            val author = readLine() ?: ""
            println("Enter year published:")
            val yearPublished = readLine()?.toIntOrNull() ?: 0
            println("Enter number of copies:")
            val copies = readLine()?.toIntOrNull() ?: 0

            val book = Book(title, author, yearPublished, copies)
            books.add(book)
        }
    }

    override fun toString(): String {
        return "\n\tTitle: $title" +
                "\n\tAuthor: $author" +
                "\n\tYear Published: $yearPublished" +
                "\n\tNumber of copies: $copies\n"
    }
}

fun main() {


    val jsonString = File("books.json").readText()
    val bookList = Json.decodeFromString<List<Book>>(jsonString)
    Book.books.addAll(bookList)

    Book.addManually()

    println(Book.books)
    println("Book version = ${Book.version}")

    println("Sorting by Title...")
    Book.books.sortBy { it.title }
    println(Book.books)

    println("CHANGING COPIES TO COPIES +1")
    Book.books.forEach { book ->
        book.copies = book.copies + 1
    }

    println("Using FOR for change all titles")
    for (i in Book.books) {
        i.title = "Changed :)"
    }
    println(Book.books)

    val finlterBooks = Book.books.filter { it.yearPublished >= 1950 }

    val filteredJson = Json.encodeToString(finlterBooks)
    File("filtered_books.json").writeText(filteredJson)

}
