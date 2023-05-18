// 19.	Квартира, дом, улица, населенный пункт.
// Кузнецов Виктор Вячеславович


import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;


interface haveGarage {
    boolean placeHaveGarage();
}

interface haveParking {
    boolean placeHaveParking();
}

abstract class Settlement {
    protected String settlementName;


    public Settlement(String settlementName) {
        this.settlementName = settlementName;
    }

    public Settlement() {
        this("Москва");
    }



    public void setSettlementName(String settlementName) {
        this.settlementName = settlementName;
    }

    public String getName() {
        return settlementName;
    }

    @Override
    abstract public String toString();

}


class Street extends Settlement {
    protected String streetName;

    public Street(String settlementName, String streetName) {
        super(settlementName);
        this.streetName = streetName;
    }

    public Street() {
        this("Москва", "Ленина");
    }

    public String getStreetName() {
        return streetName;
    }

    public void setStreetName(String streetName) {
        this.streetName = streetName;
    }

    @Override
    public String toString() {
        return "Street: \n\t" +
                "Settlement: " + settlementName +
                "\n\tStreet name: " + streetName;
    }
}

class House extends Street implements haveGarage {
    protected String numberHouse;
    protected int area;
    protected int numberFloors;
    protected boolean garage;

    public House(String settlementName, String streetName, String numberHouse, int numberFloors, int area, boolean garage) {
        super(settlementName, streetName);
        this.numberHouse = numberHouse;
        this.area = area;
        this.numberFloors = numberFloors;
        this.garage = garage;
    }

    public House() {
        super("Москва", "Московский тракт");
        this.numberHouse = "119";
        this.area = 150;
        this.numberFloors = 9;
        this.garage = true;
    }

    public String getNumberHouse() {
        return numberHouse;
    }

    public void setNumberHouse(String numberHouse) {
        this.numberHouse = numberHouse;
    }

    public void setNumberHouse() {
        this.numberHouse = numberHouse;
    }

    public int getArea() {
        return area;
    }

    public void setArea(int area) {
        this.area = area;
    }

    public int getNumberFloors() {
        return numberFloors;
    }

    public void setNumberFloors(int numberFloors) {
        this.numberFloors = numberFloors;
    }

    @Override
    public String toString() {
        return "House: " +
                "\n\tSettlement: " + settlementName +
                "\n\tStreet: " + streetName +
                "\n\tHouse number: " + numberHouse +
                "\n\tNumber floors:" + numberFloors +
                "\n\tArea: " + area +
                "\n\tGarage:" + garage;
    }

    @Override
    public boolean placeHaveGarage() {
        return garage;
    }
}

class Apartment extends Street implements haveParking {
    protected int numberApartment;
    protected boolean park;
    protected String houseNumber;

    public Apartment(String settlementName, String streetName, String houseNumber, int numberApartment, boolean park) {
        super(settlementName, streetName);
        this.numberApartment = numberApartment;
        this.park = park;
        this.houseNumber = houseNumber;
    }

    public Apartment() {
        super("Москва", "Ленина");
        this.numberApartment = 15;
        this.park = false;
    }

    public boolean isPark() {
        return park;
    }

    public void setPark(boolean park) {
        this.park = park;
    }

    public int getNumberApartment() {
        return numberApartment;
    }

    public void setNumberApartment(int numberApartment) {
        this.numberApartment = numberApartment;
    }

    @Override
    public String toString() {
        return "Apartment: " +
                "\n\tSettlement: " + settlementName +
                "\n\tStreet: " + streetName +
                "\n\tHouse number:" + houseNumber +
                "\n\tApartment number: " + numberApartment +
                "\n\tParking: " + park;
    }

    public String getHouseNumber() {
        return houseNumber;
    }

    public void setHouseNumber(String houseNumber) {
        this.houseNumber = houseNumber;
    }

    @Override
    public boolean placeHaveParking() {
        return park;
    }
}


public class Main {
    public static void main(String[] args) {

        /*var ap = new House("Тюмень", "Московский тракт", "135", 9, 150);
        var ap1 = new Apartment("Тюмень", "Московский тракт", "135", 9, 150, 15);
        System.out.println(ap);
        System.out.println(ap1);*/


        System.out.println("Свойство полиморфизма");
        Street lenina = new Street("Москва", "Ленина");
        System.out.println(lenina.toString());

        House house = new House("Москва", "Ленина", "1", 5, 100, true);
        System.out.println(house.toString());

        Apartment apartment = new Apartment("Москва", "Ленина", "1", 5, true);
        System.out.println(apartment.toString());


        System.out.println("Добавление всего в лист объектом которого является родительский класс");
        List<Settlement> data = new ArrayList<>();
        data.add(new Apartment("Екатеринбург", "Ленина", "5", 12, false));
        data.add(new Apartment("Казань", "Кремлевская", "2", 9, true));

        data.add(lenina);
        data.add(house);
        data.add(apartment);

        for (Settlement dat : data) {
            System.out.println(dat.toString());
        }

        System.out.println("Filter houses");

        List<House> dataHouses = filterHouses(data);
        for (House house1 : dataHouses) {
            System.out.println(house1);
        }
        System.out.println("Filter apartments");

        List<Apartment> dataApartments = filterApartment(data);
        for (Apartment apartment1 : dataApartments) {
            System.out.println(apartment1);
        }


        var sc = new Scanner(System.in);
        System.out.println("Введите количество этажей больше которых хотите найти (в базе только дом с 5 этажами)");
        int x = sc.nextInt();

        var dataFloors = getHousesWithFloorsMoreThanFive(data, x);
        for (var dat : dataFloors) {
            System.out.println(dat);
        }

        System.out.println("Вызов интерфейсов");
        for (Settlement settlement : data) {
            if (settlement instanceof haveGarage garage) {
                System.out.println(settlement.getName() + " " + settlement.getClass() + " has garage: " + garage.placeHaveGarage());
            }

            if (settlement instanceof haveParking parking) {
                System.out.println(settlement.getName() + " " + settlement.getClass() + " has parking: " + parking.placeHaveParking());
            }
        }
    }

    public static List<House> filterHouses(List<Settlement> list) {
        return list.stream()
                .filter(obj -> obj.getClass().getName().equals("House"))
                .map(obj -> (House) obj)
                .collect(Collectors.toList());
    }

    public static List<Apartment> filterApartment(List<Settlement> list) {
        return list.stream()
                .filter(obj -> obj instanceof Apartment)
                .map(obj -> (Apartment) obj)
                .collect(Collectors.toList());
    }

    public static List<House> getHousesWithFloorsMoreThanFive(List<Settlement> list, int x) {
        return list.stream()
                .filter(obj -> obj instanceof House && ((House) obj).getNumberFloors() > x)
                .map(obj -> (House) obj)
                .collect(Collectors.toList());
    }

}
