"use strict";

$(document).ready(function(){
    // Открыть модальное окно
    $('#showModal').click(function(){
        $('#overlay').fadeIn(300);
        $('#modal').fadeIn(300);
    });

    // Закрыть модальное окно
    $('#close').click(function(){
        $('#overlay').fadeOut(300);
        $('#modal').fadeOut(300);
    });
});


//jQuery
$(document).ready(function() {
    var $slider = $('.slider');
    var $container = $('.slider-container');
    var $slides = $container.find('img');
    var slideCount = $slides.length;
    var slideWidth = $slider.width();
    var currentIndex = 0;

    // Установка ширины контейнера со слайдами
    $container.css('width', slideCount * slideWidth);

    // Обработчик события нажатия на кнопку "Next"
    $('.slider-next').click(function() {
        if (currentIndex < slideCount - 1) {
            currentIndex++;
            $container.animate({
                'margin-left': -currentIndex * slideWidth
            }, 500);
        }
        checkButtons();
    });

    // Обработчик события нажатия на кнопку "Prev"
    $('.slider-prev').click(function() {
        if (currentIndex > 0) {
            currentIndex--;
            $container.animate({
                'margin-left': -currentIndex * slideWidth
            }, 500);
        }
        checkButtons();
    });

    // Проверка доступности кнопок "Next" и "Prev"
    function checkButtons() {
        $('.slider-prev').prop('disabled', currentIndex === 0);
        $('.slider-next').prop('disabled', currentIndex === slideCount - 1);
    }

    // Инициализация
    checkButtons();
});


//Bootsrap
$(document).ready(function() {
    var carousel = $("#carouselExampleControls").carousel({
        interval: false // Отключаем автоматическое перелистывание слайдов
    });

    // Событие, которое срабатывает перед перелистыванием слайдов
    carousel.on('slide.bs.carousel', function (event) {
        var totalItems = $('.carousel-item').length;
        var currentIndex = $('div.active').index() + 1;

        // Проверяем, является ли текущий слайд первым или третьим
        if (currentIndex === 1) {
            // Блокируем перелистывание на следующий слайд
            if (event.direction === 'right') {
                event.preventDefault();
            }
        } else if (currentIndex === totalItems) {
            // Блокируем перелистывание на предыдущий слайд
            if (event.direction === 'left') {
                event.preventDefault();
            }
        }
    });

    // Событие, которое срабатывает после перелистывания слайдов
    carousel.on('slid.bs.carousel', function (event) {
        var totalItems = $('.carousel-item').length;
        var currentIndex = $('div.active').index() + 1;

        // Проверяем, является ли текущий слайд первым или третьим
        if (currentIndex === 1) {
            // Блокируем перелистывание на предыдущий слайд
            $('.carousel-control-prev').addClass('disabled');
            $('.carousel-control-next').removeClass('disabled');
        } else if (currentIndex === totalItems - 1) {
            // Блокируем перелистывание на следующий слайд
            $('.carousel-control-next').addClass('disabled');
            $('.carousel-control-prev').removeClass('disabled');
        } else {
            // Разблокируем обе кнопки
            $('.carousel-control-prev').removeClass('disabled');
            $('.carousel-control-next').removeClass('disabled');
        }
    });
});
