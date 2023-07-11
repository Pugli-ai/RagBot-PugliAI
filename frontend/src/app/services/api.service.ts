import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private http: HttpClient) {}

  // // Define your HTTP request function
  // public makeHttpRequest(url: string, body: any): Observable<any> {
  //   // Define any headers you may need
  //   const headers = new HttpHeaders({
  //     'Content-Type': 'application/json'
  //   });

  //   return this.http.post(url, body, { headers });


  public saveUser(text:string): Observable<any> {
    const headers = new HttpHeaders({
          'Content-Type': 'application/json'
        });

    const url = 'https://test123q.free.beeceptor.com/todos';
    return this.http.post<any>(url, text, { headers });
  }
  
}
