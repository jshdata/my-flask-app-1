
function fetchTodos(){


   $(document).on('click','.todo-box',function(){

      var todoId = $(this).data('todo-id');
      location.href='http://127.0.0.1:5000/detail-todo?todoid='+todoId;
   });
   

   $.ajax({
      url:'https://jsonplaceholder.typicode.com/todos/',
      type:'get',
      data:{},
      success:function(todos){
        
        


         $.each(todos, function(index, todo){
            $('#todos').append(`
               <tr style="cursor:pointer;" class="todo-box" data-todo-id="${todo.id}">
                  <th scope="row">${todo.id}</th>
                  <td>${todo.userId}</td>
                  <td>${todo.title}</td>
                  <td>${todo.completed ? '성공' : '실패'}</td>
               </tr>
         `);


         });


        

         
      },
      error:function(error){}
   });

}

$(document).ready(function(){

   fetchTodos();

   $.ajax({
      url:'https://jsonplaceholder.typicode.com/todos/2',
      type:'get',
      data:{},
      success:function(todo){
         console.log(todo.title);
         $('#todo-id').html(todo.id);
         $('#user-id').html(todo.userId);
         $('#title').html(todo.title);
         $('#completed').html(todo.completed.toString());

      },
      error:function(error){}
   });







   var index=0;

   $('#save-btn').on('click',function(){
      var name = $('#name').val();
      var age = $('#age').val();
      var address = $('#address').val();

      console.log(name,age,address);


      //유효성 검사
      if(name.length == 0 ){
         alert('이름을 입력해주세요');
         return;
      }

      if(age==0){
         alert('나이를 입력해주세요');
         return;
      }

      if(address.length == 0){
         alert('주소를 입력해주세요');
         return;
      }
      index++;
      $('#users').append(`
               <tr>
                <th scope="row">${index}</th>
                <td>${name}</td>
                <td>${age}</td>
                <td>${address}</td>
              </tr>
      `);


      alert('등록완료');
      $('#name').val('');
      $('#age').val('');
      $('#address').val('');
     
   });


   $('#address').on('click',function(){
      new daum.Postcode({
         oncomplete: function(data) {
             // 팝업에서 검색결과 항목을 클릭했을때 실행할 코드를 작성하는 부분입니다.
             // 예제를 참고하여 다양한 활용법을 확인해 보세요.
   
   
             $('#address').val(data.address);
         }
     }).open();
   
   });

   






   $('#my-btn').on('click',function(){
      //hello -> 안녕

      var v = $('#inp').val();
      $('#txt')
      .html(v)
      .css('color','blue');
      
      $('body').css('background',v);

   });


});


