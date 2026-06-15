const CLIENT_ID = import.meta.env.VITE_OAUTH_CLIENT_ID as string;
const KEY = "kc_id_token";

export const getToken = () => sessionStorage.getItem(KEY);
export const setToken = (t: string) => sessionStorage.setItem(KEY, t);
export const clearToken = () => sessionStorage.removeItem(KEY);

export function renderSignIn(el: HTMLElement, onToken: (t: string) => void) {
  // @ts-expect-error GSI global
  google.accounts.id.initialize({
    client_id: CLIENT_ID, hosted_domain: "lemnisca.bio",
    callback: (r: { credential: string }) => { setToken(r.credential); onToken(r.credential); },
  });
  // @ts-expect-error GSI global
  google.accounts.id.renderButton(el, { theme: "outline", size: "large" });
}
