import { Component, ElementRef, ViewChild } from '@angular/core';
import { AlertService } from '../../services/alert.service';
import { CookieService } from '../../services/cookie.service';
import { ApiService } from 'src/app/services/api.service';
import { HttpClient } from '@angular/common/http';




@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {

  @ViewChild('myTextarea') myTextarea!: ElementRef;
  @ViewChild('chatContainer') chatContainer!: ElementRef;

  isSidebarOpen = true;
  isTyping = false;
  isPreviewOpen = false;
  

  history:any = []
  
  
  constructor(private as:AlertService,
              private cs:CookieService,
              private cookie: CookieService,
              private api: ApiService,
              private http: HttpClient){
              }


  chat(userText:string){

    
    this.history.push(
      {"isUser": true, "text":userText},
    );

    this.api.chat(this.myTextarea.nativeElement.value).subscribe((response: any) => {
      this.history.push(
        {"isUser": false, "text":response.answer},
      );
    });
    this.myTextarea.nativeElement.value = '';

  }


}
