## Реализовать два класса Pupa и Lupa. И класс Accountant.
Класс Accountant должен уметь одинаково успешно работать и с экземплярами класса Pupa и с экземплярами класса Lupa. 
У класса Accountant должен быть метод give_salary(worker).  
Который, получая на вход экземпляр классов Pupa или Lupa, вызывает у них метод take_salary(int).  
Необходимо придумать как реализовать такое поведение.  
Метод take_salary инкрементирует внутренний счётчик у каждого экземпляра класса на переданное ему значение.  
При этом Pupa и Lupa два датасайнтиста и должны работать с матрицами.  
У них есть метод do_work(filename1, filename2).  
Pupa считывают из обоих переданных ему файлов по матрице и поэлементно их суммируют.  
Lupa считывают из обоих переданных ему файлов по матрице и поэлементно их вычитают.   
Работники обоих типов выводят результат своих трудов на экран.
Класс Accountant реализует логику начисления ЗП на ваше усмотрение, 
но будьте внимательны чтобы не получилось так, что Lupa получит за Pupa, а Pupa ничего не получит.
