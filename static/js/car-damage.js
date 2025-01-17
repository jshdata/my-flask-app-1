$(document).ready(function() {
    // 차량 손상 예측
    $('#upload-form').on('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(this);

        // AJAX 요청
        $.ajax({
            url: './api/ai/predict-car-damage',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#damage-result').html(`예측된 손상 형태: ${response.damage}<br>예상 비용: ${response.cost}`);
            },
            error: function(xhr) {
                const errorMsg = xhr.responseJSON?.error || '요청 처리 중 오류가 발생했습니다.';
                $('#damage-result').text(`오류: ${errorMsg}`);
            }
        });
    });
});