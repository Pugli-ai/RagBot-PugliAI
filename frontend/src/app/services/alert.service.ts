import { Injectable } from '@angular/core';
import Swal from 'sweetalert2';

@Injectable({
  providedIn: 'root'
})
export class AlertService {

  constructor() { }

  errorAlert(title: string, message: string){
    return Swal.fire({
      heightAuto: false,
      customClass: 'visai-settings-form',
      confirmButtonColor: 'var(--orma-green)',
      icon: 'error',
      title: title,
      text: message
    });
  }

  addOpenAIkey(){
    return Swal.fire({
      title: "OpenAI secret key",
      text: "Insert here your OpenAI secret key. This key is not shared with us and it is stored only on your pc",
      input: 'text',
      customClass: 'visai-settings-form',
      confirmButtonText: "Save",
      confirmButtonColor: 'var(--color-primary)',
      heightAuto: false,      
      backdrop: true
    })
  }

  info(){
    return Swal.fire({
      heightAuto: false,
      customClass: 'visai-settings-form',
      confirmButtonColor: 'var(--orma-green)',
      icon: 'info',
      title: "Done!",
      text: "Operation completed correctly"
    });
  }

  csvAlert(){
    return Swal.fire({
      heightAuto: false,
      title: 'Upload dataset',
      customClass: 'visai-settings-form',
      confirmButtonColor: 'var(--orma-green)',
      html:
      '<div class="container">' +
          '<div class="row justify-content-center text-right">'+
              '<div class="col px-5">'+
                  '<div class="my-3">'+
                      '<span>Insert a link to your dataset here:</span>'+
                      '<div class="form-group my-1">'+
                          '<input id="customdataset" type="text" class="form-control">'+
                      '</div>'+
                  '</div>'+
                  '<div class="my-3">'+
                      '<span>Or select a toy dataset from the dropdown menu:</span>'+
                      '<div class="form-group my-1">'+
                         ' <select id="toydataset" class="form-control">'+
                              '<option value="https://gist.githubusercontent.com/rnirmal/e01acfdaf54a6f9b24e91ba4cae63518/raw/6b589a5c5a851711e20c5eb28f9d54742d1fe2dc/datasets.csv">Car dataset</option>'+
                              '<option value="https://gist.githubusercontent.com/rnirmal/e01acfdaf54a6f9b24e91ba4cae63518/raw/6b589a5c5a851711e20c5eb28f9d54742d1fe2dc/datasets.csv">Medical dataset</option>'+
                              '<option value="https://gist.githubusercontent.com/rnirmal/e01acfdaf54a6f9b24e91ba4cae63518/raw/6b589a5c5a851711e20c5eb28f9d54742d1fe2dc/datasets.csv">Financial dataset</option>'+
                          '</select>'+
                      '</div>'+
                  '</div>'+
              '</div>'+
          '</div>'+
      '</div>',
      confirmButtonText: 'Submit',
      preConfirm: () => {
        const dropdownValue = (<HTMLInputElement>document.getElementById('toydataset')).value;
        const textInputValue = (<HTMLInputElement>document.getElementById('customdataset')).value;        
        return { dropdownValue, textInputValue };
      }
      })
    }

    inputAlert(){
      return Swal.fire({
        heightAuto: false,
        customClass: 'visai-settings-form',
        confirmButtonColor: 'var(--color-primary)',
        confirmButtonText:'Submit',
        title: "Report content!",
        // text: "Operation completed correctly" ,
        input: 'text',
        showCancelButton: true        
      })
    }

  //   .then((result) => {
  //     if (result.value) {
  //         console.log("Result: " + result.value);
  //     }
  // });
  
}
