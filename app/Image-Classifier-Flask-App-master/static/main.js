$(document).ready(function(){

  console.log("Working")

  function readURL(input) {

  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function(e) {
      $('#image').attr('src', e.target.result);
    }

    reader.readAsDataURL(input.files[0]);
  }
}

    $("input").change(function() {
        readURL(this);
    });

});
