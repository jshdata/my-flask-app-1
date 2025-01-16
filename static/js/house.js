$(document).ready(function(){

      $('#predict-btn').click(function(){
          var area = $('#area').val();
          var rooms = $('#rooms').val();
          var year = $('#year').val();
          var income = $('#income').val();
          var school_rating = $('#school-rating').val();
          var transit_score = $('#transit-score').val();
  
          console.log(area, year, income, school_rating, transit_score, rooms)
          if (area == '') {
              alert('건축면적을 입력해주세요');
              return;
          }
          if (year == '') {
              alert('건축년도를 입력해주세요'); 
              return;
          }
          if (income == '') {
              alert('주택 가격을 입력해주세요'); 
              return;
          }
          if (school_rating == '') {
              alert('학교 점수를 입력해주세요');
              return;
          }
          if (transit_score == '') {
              alert('교통 점수를 입력해주세요');
              return;
          }
          if (school_rating < 1 || school_rating > 10) {
              alert('학교 점수는 1~10 사이로 입력해주세요');
              return;
          }
          if (transit_score < 1 || transit_score > 10) {
              alert('교통 점수는 1~10 사이로 입력해주세요');
              return;
          }
          
          $.ajax({
              url:'./api/ai/predict-house-price',
              type:'get',
              contentType:"application/json",
              data:{
                  'area':area,
                  'year':year,
                  'income':income,
                  'school_rating':school_rating,
                  'transit_score':transit_score,
                  'rooms':rooms
              },
              success:function(response){
                  $('#result-text-1').text('$' + response.price_linear_prediction.toLocaleString());
                  $('#result-text-2').text('$' + response.price_random_forest_prediction.toLocaleString());
  
                  
                  $.ajax({
                      url: 'http://127.0.0.1:5000/api/ai/add-house',
                      type: 'post',
                      contentType: "application/json",
                      data: JSON.stringify({
                          'area': area,
                          'rooms': rooms,
                          'year': year,
                          'income': income,
                          'school_rating': school_rating,
                          'transit_score': transit_score,
                          'pred_lin': parseFloat(response.price_linear_prediction),
                          'pred_rf': parseFloat(response.price_random_forest_prediction)
                      }),
                      success:function(response){
                          console.log(response);
                      },
                      error:function(error){
                          console.log(error);
                      }
                  });
              },
              error:function(error){
                  console.log(error);
              }
          });
      });
  });