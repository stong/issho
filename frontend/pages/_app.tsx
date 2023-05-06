import type { AppProps } from 'next/app'
import { MantineProvider } from "@mantine/core";
import { Notifications } from '@mantine/notifications';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <MantineProvider theme={{ colorScheme: 'dark' }} withGlobalStyles withNormalizeCSS>
      <Notifications />
      <Component {...pageProps} />
    </MantineProvider>
  );
}
