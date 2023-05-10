import Head from 'next/head'

import styles from '@/styles/Home.module.css'

import { useState, useEffect, useRef } from 'react';

import { Box, Center, Text, Image } from "@mantine/core";

import { notifications } from '@mantine/notifications';
import { IconX, IconCheck } from '@tabler/icons-react';


import { Inter } from 'next/font/google'
const inter = Inter({ subsets: ['latin'] })

function Top() {
  return (
    <>
      <Center pb={10}>
        <Text size="xl" fw={700} py="lg">
          {"Issho"}
        </Text>
        {<Image src="anime.png" alt="anime" width={48} height={48} />}
      </Center>
    </>
  );
}

import { Textarea, Button, Group, Stack, Accordion, Slider, Grid, Space, Select } from "@mantine/core";

function useLocalStorage(key: string, defaultValue: any) {
  const [value, setValue] = useState(defaultValue);

  useEffect(() => {
    const storedValue = localStorage.getItem(key);
    if (storedValue) {
      try {
        setValue(JSON.parse(storedValue));
      } catch(error) {
        console.log("Invalid json: " + key);
      }
    }
  }, []);

  useEffect(() => {
    if (value !== defaultValue) {
      localStorage.setItem(key, JSON.stringify(value));
    }
  }, [value]);

  return [value, setValue];
}

function Textgen() {
  const [value, setValue] = useLocalStorage("prompt", "");
  const [cooldownTimer, setCooldownTimer] = useState(0);
  const [temperature, setTemperature] = useLocalStorage("temperature", 1.99);
  const [topP, setTopP] = useLocalStorage("topP", 0.29);
  const [topK, setTopK] = useLocalStorage("topK", 30);
  const [maxTokens, setMaxTokens] = useLocalStorage("maxTokens", 200);
  const [wasError, setWasError] = useState(false);
  const [backendState, setBackendState] = useState('idle');
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const websocketRef = useRef(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
  };

  // this is a dumb hack
  useEffect(() => {
    (async () => {
      if (localStorage.getItem("prompt") === null) {
        const resp = await fetch("/prompt.txt");
        const text = await resp.text();
        setValue(text);
      }
    })();
  }, []);

  // this too
  const [didInitialScroll, setDidInitialScroll] = useState(false);
  useEffect(() => {
    if (value != "" && !didInitialScroll) {
      scrollToBottom();
      setDidInitialScroll(true);
    }
  }, [value]);

  const connectWebsocket = () => {
    const currentUrl = new URL(window.location.href);
    currentUrl.protocol = currentUrl.protocol == 'https:' ? 'wss:' : 'ws:';
    currentUrl.pathname = '/ws';
    const socketUrl = currentUrl.toString();
    console.log(socketUrl);
    websocketRef.current = new WebSocket(socketUrl);

    const handleOpen = (event) => {
      console.log('WebSocket connection opened:', event);
      setConnectionStatus('connected');
      notifications.show({
        title: 'Server status',
        message: 'Connected',
        icon: <IconCheck />,
      })
    };

    const handleMessage = (event) => {
      console.log('WebSocket message received:', event);
      const parsedData = JSON.parse(event.data);
      if (parsedData.status === "set_state") {
        setBackendState(parsedData.new_state);
      } else if (parsedData.status === "error") {
        setWasError(true);
        notifications.show({
          title: 'Error',
          message: parsedData.message
        });
      } else if (parsedData.status === "progress") {
        setValue(parsedData.completion);
        scrollToBottom();
      } else if (parsedData.status === "notification") {
        notifications.show({
          title: 'Notification',
          message: parsedData.message
        })
      } else if (parsedData.status === "cooldown_timer") {
        setCooldownTimer(parsedData.seconds_remaining);
      }
    };

    const handleClose = (event) => {
      console.log('WebSocket connection closed:', event);
      setBackendState('idle');
      if (backendState === 'connected') {
        notifications.show({
          title: 'Server status',
          message: 'Disconnected',
          icon: <IconX />,
          color: 'red'
        });
      }
      setConnectionStatus('disconnected');
      setTimeout(connectWebsocket, 5000);
    };

    const handleError = (event) => {
      websocketRef.current.close();
      console.error('WebSocket error:', event);
      setConnectionStatus('disconnected');
      setBackendState('idle');
    };

    websocketRef.current.addEventListener('open', handleOpen);
    websocketRef.current.addEventListener('message', handleMessage);
    websocketRef.current.addEventListener('close', handleClose);
    websocketRef.current.addEventListener('error', handleError);
  };

  useEffect(() => {
    connectWebsocket();

    // Clean up the WebSocket connection when the component is unmounted
    return () => {
      websocketRef.current.close();
    };
  }, []);

  function doGenerate(event) {
    // this is kinda a ui hack but w.e
    setBackendState('busy');
    setCooldownTimer(10);

    setWasError(false);
    if (websocketRef.current) {
      websocketRef.current.send(JSON.stringify({
        'action': 'generate',
        'prompt': value,
        'max_tokens': maxTokens,
        'temperature': temperature,
        'top_p': topP,
        'top_k': topK,
      }));
    }
  }

  function doStop(event) {
    if (websocketRef.current) {
      websocketRef.current.send(JSON.stringify({'action': 'cancel'}));
    }
  }

  function isGenerating() {
    return backendState === 'busy' && connectionStatus === 'connected';
  }

  return (
    <>
      <Stack>
        <Textarea
          minRows={20}
          placeholder={"Your prompt\n\nRemember, generated text is made up!"}
          value={value}
          error={wasError}
          onChange={(event) => setValue(event.currentTarget.value)}
          ref={textareaRef}
        />

        <Group grow>
          <Button
            loading={backendState === 'busy'}
            disabled={backendState !== 'idle' || connectionStatus !== 'connected'}
            onClick={doGenerate}
          >
            {connectionStatus === 'connected' ?
              backendState === 'cooldown' ? `Please wait (${cooldownTimer})` : "Generate"
              : "Disconnected"
            }
          </Button>
          <Button
            disabled={!isGenerating()}
            onClick={doStop}
          >
            Stop
          </Button>
        </Group>

        <Box>
          <Text size="sm">Output Size Limit</Text>
          <Slider
            min={20}
            max={400}
            marks={[
              { value: 20, label: "20" },
              { value: 200, label: "200" },
              { value: 400, label: "400" },
            ]}
            value={maxTokens}
            onChange={setMaxTokens}
            step={1}
            disabled={isGenerating()}
          />
        </Box>

        <Space h="md" />

        <Accordion defaultValue="privacy">
          <Accordion.Item value="advancedSettings">
            <Accordion.Control>Generation Settings</Accordion.Control>
            <Accordion.Panel>
              <Grid py="md">
                <Grid.Col span={6} py="md">
                  <Box>
                    <Text size="sm">Temperature</Text>
                    <Slider
                      labelAlwaysOn
                      min={0}
                      max={2}
                      marks={[
                        { value: 0, label: "0" },
                        { value: 1, label: "1" },
                        { value: 2, label: "2" },
                      ]}
                      value={Number((Math.round(temperature * 100) / 100).toFixed(2))}
                      onChange={setTemperature}
                      step={0.01}
                      disabled={isGenerating()}
                    />
                  </Box>
                </Grid.Col>
                <Grid.Col span={6} py="md">
                  <Box>
                    <Text size="sm">Top P</Text>
                    <Slider
                      labelAlwaysOn
                      min={0.01}
                      max={1}
                      marks={[
                        { value: 0.01, label: "0" },
                        { value: 1, label: "1" },
                      ]}
                      value={Number((Math.round(topP * 100) / 100).toFixed(2))}
                      onChange={setTopP}
                      step={0.01}
                      disabled={isGenerating()}
                    />
                  </Box>
                </Grid.Col>
                <Grid.Col span={6} py="md">
                  <Box>
                    <Text size="sm">Top K</Text>
                    <Slider
                      labelAlwaysOn
                      min={0}
                      max={100}
                      marks={[
                        { value: 0, label: "0" },
                        { value: 100, label: "100" },
                      ]}
                      value={topK}
                      onChange={setTopK}
                      step={1}
                      disabled={isGenerating()}
                    />
                  </Box>
                </Grid.Col>
                <Grid.Col span={6}>
                  <Select
                    disabled
                    label="Model"
                    value="Model"
                    data={[{ value: 'Model', label: 'Model' }]}
                  />
                </Grid.Col>
              </Grid>
            </Accordion.Panel>
          </Accordion.Item>
          <Accordion.Item value="privacy">
            <Accordion.Control>Privacy</Accordion.Control>
            <Accordion.Panel>
            <Text>- Prompts are only stored when it is in the queue and are deleted immediately after generation.</Text>
            <Text>- Generated text is NOT stored on the server. We do NOT store your generations.</Text>
            <Text>- We do not share your data.</Text>
            <Text>- You are welcome to use privacy enhancing technologies like VPNs or Tor.</Text>
            <Text>- <a href="https://github.com/stong/issho">Source code</a></Text>
            </Accordion.Panel>
          </Accordion.Item>
          <Accordion.Item value="terms">
            <Accordion.Control>Terms of Use</Accordion.Control>
            <Accordion.Panel>
            <Text>By using this service, you agree to abide by the following terms:</Text>
            <Text>- You must be 18 years of age to use the service. You must not be legally prohibited from using the service.</Text>
            <Text>- You shall not use the service to deliberately generate harmful or illegal outputs or content.</Text>
            <Text>- You must use the service in accordance with applicable U.S. and local laws.</Text>
            <Text>- You shall not represent the generated content as non-fiction. You acknowledge that all generated content is made up.</Text>
            <Text>- All characters / persons in the generated content must be 18 years of age or older.</Text>
            <Text>We reserve the right to revoke access to the service at any time, in particular for violations of these terms.</Text>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>
    </>
  )
}

export default function Home() {
  return (
    <>
      <Head>
        <title>Issho</title>
        <meta name="description" content="An open-source creative writing assistant!" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta property="og:image" content="https://issho.ai/anime.png" />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content="Issho" />
        <meta name="twitter:description" content="An open-source creative writing assistant!" />
        <meta name="twitter:image" content="https://issho.ai/anime.png" />
        <link rel="icon" href="/favicon.ico" />
        <script defer data-domain="issho.ai" src="https://plausible.io/js/script.js"></script>
      </Head>
      <Box mx="auto" my="lg" maw="120ch" px="sm">
        <main>
          <Top />
          <Textgen />
        </main>
      </Box>
    </>
  )
}
