import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { entrypoint } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private http: HttpClient) {}

  public chat(text:string): Observable<any> {
    const headers = new HttpHeaders({
          'Content-Type': 'application/json'
        });

    const url = entrypoint + 'qa';


    let body = JSON.stringify({
      "question": text
    });
    
    console.log(body);
    return this.http.post<any>(url, body, { headers });
  }
  
}
