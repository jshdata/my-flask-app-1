$(document).ready(function(){
    //var todo_id = $('#todo-id').val();
    

    $('#study-btn').click(function(){
        $.ajax({
            url:'http://127.0.0.1:5000/api/user/study',
            type:'get',
            success:function(data){
                console.log(data);
            }
        });
    });

   $.ajax({
      url:'http://127.0.0.1:5000/api/user/saveUser',
      type:'post',
      data:{
         id:'coding',
         pw:'1234'
      },
      success:function(data){
         if(data == 'ok'){
            alert('회원가입 성공');
         }else{
            alert('회원가입 실패');
         }
      },
      error:function(error){
         console.log(error);
      }
   });








    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const todo_id = urlParams.get('todoid');



   $.ajax({
      url: 'https://jsonplaceholder.typicode.com/todos/'+todo_id,
      type: 'get',
      data: {},
      success:function(todo){
         
         $('#id').text(todo.id);
         $('#userid').text(todo.userId);
         $('#title').text(todo.title);
         $('#completed').text(todo.completed ? '성공' : '실패');
      },
      error:function(error){
         console.log(error);
      }
   });

});