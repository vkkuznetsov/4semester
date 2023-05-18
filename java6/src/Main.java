import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;


class Main {
    public static Scanner sc = new Scanner(System.in);
    public static Library mainLibrary = new Library("Main Library");

    public static void main(String[] args) {
        generateWhenStarts(); // генерирует при старте чтобы сразу можно было выполнять методы
        boolean check = true;

        while (check) {
            System.out.println("""
                    Меню
                    \t1. Записать данные класса книга вручную
                    \t2. Записать данные класса книга сгенерированными значениями
                    \t3. filter (если год больше 1000)
                    \t4. map
                    \t5. peek
                    \t6. получить первые N коллекции
                    \t7. получить последние N коллекции
                    \t8. преобразовать коллекцию в массив
                    \t9. найти в коллекции объект с минимальным или максимальным значением года выпуска
                    \t10. проверить что один из объектов коллекции удовлетворяет условию\s
                    \t11. проверить что каждый из объектов коллекции удовлетворяет условию
                    \t12. Вывод информации о всех книгах
                    \t13. Завершить работу программы.
                    """);

            int userInput = sc.nextInt();
            switch (userInput) {
                case 1 -> {
                    addNewBookManually();
                    showAllBooks(mainLibrary.getBooks());
                }
                case 2 -> {
                    generateRandomly();
                    showAllBooks(mainLibrary.getBooks());
                }

                case 3 -> {

                    ArrayList<Book> case2 = mainLibrary.filterAPI();
                    showAllBooks(case2);
                }
                case 4 -> {
                    List<String> case4 = mainLibrary.mapAPI();
                    System.out.println(case4);
                }
                case 5 -> { // увеличивает количество копий книг на 10
                    mainLibrary.peekAPI(book -> book.setCopies(book.getCopies() + 10));
                }
                case 6 -> {
                    System.out.print("Enter N = ");
                    int N = sc.nextInt();

                    ArrayList<Book> case6 = (ArrayList<Book>) mainLibrary.getFirstN(N);
                    showAllBooks(case6);
                }
                case 7 -> {
                    System.out.print("Enter N = ");
                    int N = sc.nextInt();

                    ArrayList<Book> case7 = (ArrayList<Book>) mainLibrary.getLastN(N);
                    showAllBooks(case7);
                }
                case 8 -> {
                    Book[] case8  = mainLibrary.toArray();
                    for (int i = 0; i < case8.length; i++) {
                        System.out.println(case8[i].toString());
                    }
                }
                case 9 -> {
                    System.out.println("Вывод книги с максимальным - 1 минимальным - 2 количеством копий");
                    int ans = sc.nextInt();
                    if (ans == 1){
                        Book case9max = mainLibrary.findBookWithMaxCopies();
                        System.out.println(case9max.toString());
                    }
                    else if (ans == 2){
                        Book case9min = mainLibrary.findBookWithMinCopies();
                        System.out.println(case9min.toString());
                    }
                }
                case 10 -> {
                    sc.nextLine(); // если это не сделать, то проблемы с буфером ввода
                    System.out.println("Введите название книги которую хотите проверить (Test generated23) - точно есть, чтобы проверить что нет пишите 3 цифры");
                    String title = sc.nextLine();
                    System.out.println(mainLibrary.hasBookWithTitle(title));

                }
                case 11 -> {
                    System.out.println("Введите количесвто копий = (Выведет true если у всех книг больше чем это значение)");
                    int n = sc.nextInt();

                    System.out.println(mainLibrary.allBooksHaveMoreCopiesThan(n));
                }
                case 12 -> {
                    showAllBooks(mainLibrary.getBooks());
                }
                case 13 -> {
                    System.out.println("Завершение работы программы");
                    check = false;
                }


            }
        }
    }

    private static void showAllBooks(ArrayList<Book> books) {

        for (int i = 0; i < books.size(); i++) {
            System.out.printf("Book #%d", i + 1);
            System.out.print(books.get(i));
        }
    }

    private static void generateWhenStarts() {
        Random rd = new Random();
        for (int i = 1; i <= 30; i++) {
            String title = "Test generated" + i;
            String author = "Name author" + i;
            int yearPublished = rd.nextInt(1, 2023);
            int copies = rd.nextInt(1, 200);
            Book book = new Book(title, author, yearPublished, copies);
            mainLibrary.addBook(book);
        }
    }

    private static void generateRandomly() {
        Random rd = new Random();
        System.out.print("Введите сколько хотите сгенерировать данных = ");
        int N = sc.nextInt();
        for (int i = 1; i < N + 1; i++) {
            String title = "Test generated" + i;
            String author = "Name author" + i;
            int yearPublished = rd.nextInt(1, 2023);
            int copies = rd.nextInt(1, 200);
            Book book = new Book(title, author, yearPublished, copies);
            mainLibrary.addBook(book);
        }
    }


    private static void addNewBookManually() {
        System.out.print("Введите название книги = ");
        String title = sc.next();
        System.out.print("Введите Автора книги = ");
        String author = sc.next();
        System.out.print("Введите год издания книги книги (формат yyyy)= ");
        int yearPublished = sc.nextInt();
        System.out.print("Введите количество копий книг = ");
        int copies = sc.nextInt();
        Book book = new Book(title, author, yearPublished, copies);
        mainLibrary.addBook(book);
    }

}

class Book {
    private String title;
    private String author;
    private int yearPublished;
    private int copies;

    public Book(String title, String author, int yearPublished, int copies) {
        this.title = title;
        this.author = author;
        this.yearPublished = yearPublished;
        this.copies = copies;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public int getYearPublished() {
        return yearPublished;
    }

    public int getCopies() {
        return copies;
    }

    public void setCopies(int copies) {
        this.copies = copies;
    }


    @Override
    public String toString() {
        return "\n\tTitle: " + getTitle()
                + "\n\tAuthor: " + getAuthor()
                + "\n\tYear Published: " + getYearPublished()
                + "\n\tNumber of copies: " + getCopies() + "\n";
    }

}

class Library {

    private String name;
    private ArrayList<Book> books;

    public Library(String name) {
        this.name = name;
        this.books = new ArrayList<Book>();
    }

    public String getName() {
        return name;
    }

    public ArrayList<Book> getBooks() {
        return books;
    }

    public void addBook(Book book) {
        books.add(book);
    }

    public void removeBook(Book book) {
        books.remove(book);
    }

    public ArrayList<Book> filterAPI() {
        // берет книги, фильтрует по тому больше ли год чем 1000, поток переводит в лист который приводится к arraylist<book>
        return (ArrayList<Book>) books.stream().filter(book -> 1000 < book.getYearPublished()).collect(Collectors.toList());
    }
    public List<String> mapAPI(){
        return books.stream().map(Book::toString).collect(Collectors.toList());
    }

    public void peekAPI(Consumer<Book> bookConsumer) {
        books.stream().peek(bookConsumer).forEach(book -> book.setCopies(book.getCopies()));
    }

    public List<Book> getFirstN(int n) {
        return books.stream()
                .limit(Math.min(n, books.size()))
                .collect(Collectors.toList());
    }

    public List<Book> getLastN(int n) {
        int startIndex = Math.max(books.size() - n, 0);
        return books.stream()
                .skip(startIndex)
                .collect(Collectors.toList());
    }

    public Book[] toArray() {
        return books.stream()
                .toArray(Book[]::new);
    }
    public Book findBookWithMaxCopies() {
        return books.stream()
                .max(Comparator.comparing(Book::getCopies))
                .orElse(null);
    }

    public Book findBookWithMinCopies() {
        return books.stream()
                .min(Comparator.comparing(Book::getCopies))
                .orElse(null);
    }
    public boolean hasBookWithTitle(String title) {
        return books.stream()
                .anyMatch(book -> book.getTitle().equals(title));
    }
    public boolean allBooksHaveMoreCopiesThan(int numberOfCopies) {
        return books.stream().allMatch(book -> book.getCopies() > numberOfCopies);
    }

}

