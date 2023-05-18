import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;



class Main {
    public static Scanner sc = new Scanner(System.in);
    public static Library mainLibrary = new Library("Main Library");

    public static void main(String[] args) {

        boolean check = true;

        while (check) {
            System.out.println("""
                    Меню
                    \t1.Записать данные класса книга вручную
                    \t2.Записать данные класса книга из файла
                    \t3.Записать данные класса книга сгенерированными значениями"
                    \t4.Вывод на экран информации о книге если количество экземпляров превышает значение,
                    \tвведенное с клавиатуры, если таких книг нет, вывести соответствующее сообщение
                    \t5.Сортировку списка книг по названию и вывод отсортированного списка.
                    \t6.Вывод информации о всех книгах
                    \t7.Завершить работу программы.
                    """);

            int userInput = sc.nextInt();
            switch (userInput) {
                case 1 -> {
                    addNewBookManually();
                    showAllBooks();
                }
                case 2 -> {
                    readDataFromFile();
                    showAllBooks();
                }
                case 3 -> {
                    generateRandomly();
                    showAllBooks();
                }
                case 4 -> searchNumberCopies();
                case 5 -> {
                    System.out.println("Как хотите отсортировать список");
                    System.out.println("•\t1.Реализовав для класса интерфейс Comparable (тогда в метод sort коллекции в качестве компаратора передаётся null или использовать Collections.sort без параметра);\n" +
                            "•\t2.Реализовав класс компаратора или использовать реализацию метода при инстанцировании (создании экземпляра);\n" +
                            "•\t3.Определив компаратор с использованием лямбда выражения (т.к. интерфейс Comparator является функциональным)\n");
                    int choice = sc.nextInt();
                    switch (choice) {
                        case 1 -> {
                            mainLibrary.sortByName();
                            showAllBooks();
                        }
                        case 2 -> {
                            mainLibrary.sortByName2();
                            showAllBooks();
                        }
                        case 3 -> {
                            mainLibrary.sortByName3();
                            showAllBooks();
                        }
                        default -> System.out.println("Нажми нормально. Тут написано 1 или 2 или 3");
                    }
                }
                case 6 -> showAllBooks();
                case 7 -> {
                    System.out.println("Завершение работы программы");
                    check = false;
                }
                case 8 ->{
                   mainLibrary.filterByName().forEach(book -> System.out.println(book.toString()));

                }
            }
        }
    }

    private static void showAllBooks() {

        for (int i = 0; i < mainLibrary.getBooks().size(); i++) {
            System.out.printf("Book #%d", i + 1);
            System.out.print(mainLibrary.getBooks().get(i));
        }
    }

    private static void searchNumberCopies() {
        boolean flag = false;
        System.out.println("Введите количество экземпляров кинг = ");
        int N = sc.nextInt();
        for (int i = 0; i < mainLibrary.getBooks().size(); i++) {
            flag = true;
            if (N < mainLibrary.getBooks().get(i).getCopies()) {
                System.out.println(mainLibrary.getBooks().get(i));
            }
        }
        if (!flag) {
            System.out.println("Таких кинг не существует!");
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
            mainLibrary.addBook(book);
        }
    }

    private static void readDataFromFile() {
        System.out.println("Начинаю считывание данных из файла data.txt...");
        try {
            FileReader fr = new FileReader("data.txt");
            BufferedReader br = new BufferedReader(fr);
            String line = br.readLine();
            while (line != null) {
                StringTokenizer st = new StringTokenizer(line, ",");
                String title = st.nextToken();
                String author = st.nextToken();
                int yearPublished = Integer.parseInt(st.nextToken());
                int copies = Integer.parseInt(st.nextToken());

                Book book = new Book(title, author, yearPublished, copies);
                mainLibrary.addBook(book);
                line = br.readLine();
            }
            System.out.println("Данные успешно считаны");
            br.close();
            fr.close();
        } catch (Exception e) {
            e.printStackTrace();
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

class Book implements Comparable<Book>{
    private String title;
    private String author;
    private int yearPublished;
    private int copies;
    @Override
    public int compareTo(Book other){
        return title.compareTo(other.getTitle());
    }
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
    public void sortByName(){
        Collections.sort(getBooks());
    }
    public void sortByName2(){
        Comparator<Book> comparator = new BookComparator();
        Collections.sort(books, comparator);
        // можно написать books.sort(comparator); будет тоже самое
    }
    Comparator<Book> titleComparator = (b1, b2) -> b1.getTitle().compareTo(b2.getTitle());
    public void sortByName3(){
        books.sort(titleComparator);
    }

    public List<Book> filterByName(){
        return books.stream().filter(book -> Character.isDigit(book.getTitle().charAt(0))).toList();
    }
}
class BookComparator implements Comparator<Book>{
    @Override
    public int compare(Book b1, Book b2){
        return b1.getTitle().compareTo(b2.getTitle());
    }
}
