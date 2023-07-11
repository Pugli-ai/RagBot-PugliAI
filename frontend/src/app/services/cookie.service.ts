import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class CookieService {

  constructor() { }

  // Set a cookie with the given name, value, and expiration in days
  setCookie(name: string, value: string, expirationDays: number): void {
    const date = new Date();
    date.setTime(date.getTime() + (expirationDays * 24 * 60 * 60 * 1000));
    const expires = `expires=${date.toUTCString()}`;
    document.cookie = `${name}=${value}; ${expires}; path=/`;
  }

  // Get the value of the cookie with the given name
  getCookie(name: string): string {
    const cookieName = `${name}=`;
    const cookieArray = document.cookie.split(';');
    for (let i = 0; i < cookieArray.length; i++) {
      let cookie = cookieArray[i];
      while (cookie.charAt(0) === ' ') {
        cookie = cookie.substring(1);
      }
      if (cookie.indexOf(cookieName) === 0) {
        return cookie.substring(cookieName.length, cookie.length);
      }
    }
    return '';
  }

  // Delete the cookie with the given name
  deleteCookie(name: string): void {
    document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
  }

  storeCookie(name:string, val:string): boolean{

    if(this.getCookie(name) !== '')
      this.deleteCookie(name);
    
    this.setCookie(name, val, 60);

    
    if(this.getCookie(name) !== '')
      return true
    
    return false
    
  }

}
