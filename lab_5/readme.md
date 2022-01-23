## Лабораторная работа 5. Pandas.
Художественный фильм Титаник режиссера Джеймса Кэмерона славен тем, что идёт три часа и фильме кругом вода. В этой связи многие зрители покидают кинозал для посещения уборной.  
В некотором недалеком будущем 7D кинотеатр “ДК Академия” запускает показ ремастера фильма в формате 7D GALACTIC HD. И нанимает специалиста по обработке данных и машинному обучению, чтобы он рассчитал нагрузку на туалетные комнаты во время сеанса. Этот специалист вы! В первую очередь вам необходимо отфильтровать и должным образом подготовить данные, которые вам предоставил кинотеатр. За работу!  
![alt text](https://github.com/d-pack/LessonsPAK/raw/079c14e5db7f0d5c4e53b563cc0984135643899e//images/LessonsI/Cinema7D.png)  

Данные, которые предоставил кинотеатр находятся в файлах cinema_sessions.csv и titanic_with_labels  
 * Пол (sex): отфильтровать строки, где пол не указан, преобразовать оставшиеся в число 0/1;
 * Номер ряда в зале (row_number): заполнить вместо NAN максимальным значением ряда;
 * Количество выпитого в литрах (liters_drunk): отфильтровать отрицательные значения и нереально большие значения (выбросы). Вместо них заполнить средним.

 * Возраст (age): разделить на 3 группы: дети (до 18 лет), взрослые (18 - 50), пожилые (50+). закодировать в виде трёх столбцов с префиксом age_. Старый столбец с age удалить;
 * Напиток (drink): преобразовать в число 0/1 был ли этот напиток хмельным;
 * Номер чека (check_number): надо сопоставить со второй таблицей со временем сеанса. И закодировать в виде трёх столбцов, был ли это утренний (morining) сеанс, дневной (day) или вечерний (evening).
