import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.WeakHashMap;

public class Practic6 {
    public static Scanner sc = new Scanner(System.in);
    static Library library = new Library("Library for 6 practical");
    public static void main(String[] args) {
        generateRandomly();
        showAllBooks();
        boolean check = true;
        while (check){
            System.out.println("""
                    Меню
                    1. Отфильтровать коллекцию по произвольному критерию, используя filter 
                    """);


        }


    }

    private static void generateRandomly() {
        Random rd = new Random();
        System.out.print("Введите сколько хотите сгенерировать данных = ");
        int N = sc.nextInt();
        for (int i = 1; i < N + 1; i++) {
            String title = "Test generateed" + i;
            String author = "Name author" + i;
            int yearPublished = rd.nextInt(1, 2023);
            int copies = rd.nextInt(1, 200);
            Book book = new Book(title, author, yearPublished, copies);
            library.addBook(book);
        }
    }
    private static void showAllBooks() {
        for (int i = 0; i < library.getBooks().size(); i++) {
            System.out.printf("Book #%d", i + 1);
            System.out.print(library.getBooks().get(i));
        }
    }
}
