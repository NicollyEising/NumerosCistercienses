import React, { useState } from 'react';
import {
    Container, Header, Form, Button, Input, Divider, Message,
    Menu, Segment, Grid, GridColumn, Label, Image, Dimmer, Loader
} from 'semantic-ui-react';


function UploadImageArabic() {
    const [fileReconhecimento, setFileReconhecimento] = useState(null);
    const [fileConversao, setFileConversao] = useState(null);
    const [numeroReconhecido, setNumeroReconhecido] = useState('');
    const [numeroDigitado, setNumeroDigitado] = useState('');
    const [respostaNumero, setRespostaNumero] = useState('');
    const [numeroExtraido, setNumeroExtraido] = useState('');
    const [imagens, setImagens] = useState({});
    const [loading, setLoading] = useState(false);


    const resetFileInput = (selector) => {
        const input = document.querySelector(selector);
        if (input) input.value = null;
    };
    

    const handleReconhecimentoSubmit = async (e) => {
        e.preventDefault();
        if (!fileReconhecimento) return;

        const formData = new FormData();
        formData.append('imagem', fileReconhecimento);

        try {
            const response = await fetch('http://localhost:8000/upload-image-arabic', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                alert(error.detail || 'Erro ao processar a imagem.');
                return;
            }

            const data = await response.json();
            setNumeroReconhecido(data.numero);
            setImagens(data.imagens);
        } catch (error) {
            console.error('Erro:', error);
        }
    };

    const handleDigitadoSubmit = async (e) => {
        e.preventDefault();
        if (!numeroDigitado) return;

        if (numeroDigitado.length > 4) {
            alert("O número digitado não pode ter mais de 4 dígitos.");
            return;
        }
    

        try {
            const response = await fetch('http://localhost:8000/convertString', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ numero: numeroDigitado })
            });

            if (!response.ok) {
                const error = await response.json();
                alert(error.detail || 'Erro ao converter número.');
                return;
            }

            const data = await response.json();
            setRespostaNumero(data.numero);
            setImagens(data.imagens);
        } catch (error) {
            console.error('Erro:', error);
        }
    };

    const handleConversaoImagemSubmit = async (e) => {
        e.preventDefault();
        if (!fileConversao) return;

        const formData = new FormData();
        formData.append('imagem', fileConversao);

        try {
            const response = await fetch('http://localhost:8000/arabic-to-cistercian', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                alert(error.detail || 'Erro ao converter imagem.');
                return;
            }

            const data = await response.json();
            setNumeroExtraido(data.numero);
            setImagens(data.imagens);
        } catch (error) {
            console.error('Erro:', error);
        }
    };

    const handleClearAll = () => {
        setFileReconhecimento(null);
        setFileConversao(null);
        setNumeroReconhecido('');
        setNumeroDigitado('');
        setRespostaNumero('');
        setNumeroExtraido('');
        setImagens({});
        resetFileInput('#input-reconhecimento');
        resetFileInput('#input-conversao');
    };


    const [activeItem, setActiveItem] = useState('Imagem Cisterciense');

    const handleItemClick = (e, { name }) => {
        setActiveItem(name);
    };

    return (
        <Container style={{ marginTop: '2em' }}>
            <Menu pointing secondary>
                <Menu.Item
                    name="Imagem Cisterciense"
                    active={activeItem === 'Imagem Cisterciense'}
                    onClick={handleItemClick}
                />
                <Menu.Item
                    name="Digitar Número Cisterciense"
                    active={activeItem === 'Digitar Número Cisterciense'}
                    onClick={handleItemClick}
                />
                <Menu.Item
                    name="Imagem Arábica"
                    active={activeItem === 'Imagem Arábica'}
                    onClick={handleItemClick}
                />
            </Menu>


            {activeItem === 'Imagem Cisterciense' && (
    <>
        <Header as="h2">Reconhecer Número Arábico a partir de Imagem Cisterciense</Header>
        <Form onSubmit={handleReconhecimentoSubmit}>
            <Form.Field>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                        id="input-reconhecimento"
                        type="file"
                        accept="image/*"
                        onChange={(e) => setFileReconhecimento(e.target.files[0])}
                        style={{ flex: 1, marginRight: '1rem' }} // O campo de arquivo ocupará o espaço restante
                    />
                    <Button type="submit">Enviar</Button>
                </div>
            </Form.Field>
        </Form>
    </>
)}

{activeItem === 'Digitar Número Cisterciense' && (
    <>
        <Header as="h2">Converter Número Arábico Digitado para Cisterciense</Header>
        <Form onSubmit={handleDigitadoSubmit}>
            <Form.Field>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Input
                        value={numeroDigitado}
                        onChange={(e) => setNumeroDigitado(e.target.value)}
                        placeholder="Digite um número"
                        style={{ flex: 1, marginRight: '1rem' }} // Faz o Input ocupar todo o espaço restante
                    />
                    <Button type="submit">Converter</Button>
                </div>
            </Form.Field>
        </Form>
        {respostaNumero && <Message content={`Número convertido: ${respostaNumero}`} />}
    </>
)}

{activeItem === 'Imagem Arábica' && (
    <>
        <Header as="h2">Converter Imagem com Número Arábico para Cisterciense</Header>
        <Form onSubmit={handleConversaoImagemSubmit}>
            <Form.Field>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                        id="input-conversao"
                        type="file"
                        accept="image/*"
                        onChange={(e) => setFileConversao(e.target.files[0])}
                        style={{ flex: 1, marginRight: '1rem' }} // O campo de arquivo ocupará o espaço restante
                    />
                    <Button type="submit">Converter</Button>
                </div>
            </Form.Field>
        </Form>

        {numeroExtraido && <Message content={`Número extraído: ${numeroExtraido}`} />}
    </>
)}


<Button color="red" onClick={handleClearAll} style={{ marginTop: '15px', marginBottom:'50px'}}>
    Limpar Tudo
</Button>


<Grid stackable centered columns={4} style={{ marginTop: '10px' }}>
    {Object.keys(imagens).length > 0 && (
        <>
            {Object.keys(imagens).length <= 4 ? (
                <Grid.Row columns={2}>
                    {Object.entries(imagens).map(([key, value]) => (
                        <Grid.Column key={key} textAlign="center">
                            <Segment className="center aligned" padded>
                                <Label attached="top">{key}</Label>
                                <Image
                                    src={`data:image/png;base64,${value}`}
                                    alt={key}
                                    style={{
                                        maxWidth: '100%', // A imagem vai ocupar no máximo o tamanho do seu contêiner
                                        height: 'auto',   // Mantém a proporção original da imagem
                                        display: 'block', // Evita espaços extras abaixo da imagem
                                        margin: '0 auto', // Centraliza a imagem
                                    }}
                                />
                            </Segment>
                        </Grid.Column>
                    ))}
                </Grid.Row>
            ) : Object.keys(imagens).length === 5 ? (
                <>
                    <Grid.Row columns={1}>
                        {Object.entries(imagens).slice(0, 1).map(([key, value]) => (
                            <Grid.Column key={key} textAlign="center">
                                <Segment className="center aligned" padded>
                                    <Label attached="top">{key}</Label>
                                    <Image
                                        src={`data:image/png;base64,${value}`}
                                        alt={key}
                                        style={{
                                            maxWidth: '100%',
                                            height: 'auto',
                                            display: 'block',
                                            margin: '0 auto',
                                        }}
                                    />
                                </Segment>
                            </Grid.Column>
                        ))}
                    </Grid.Row>
                    <Grid.Row columns={4}>
                        {Object.entries(imagens).slice(1).map(([key, value]) => (
                            <Grid.Column key={key} textAlign="center">
                                <Segment className="center aligned" padded>
                                    <Label attached="top">{key}</Label>
                                    <Image
                                        src={`data:image/png;base64,${value}`}
                                        alt={key}
                                        style={{
                                            maxWidth: '100%',
                                            height: 'auto',
                                            display: 'block',
                                            margin: '0 auto',
                                        }}
                                    />
                                </Segment>
                            </Grid.Column>
                        ))}
                    </Grid.Row>
                </>
            ) : (
                <>
                    <Grid.Row columns={2}>
                        {Object.entries(imagens).slice(0, 2).map(([key, value]) => (
                            <Grid.Column key={key} textAlign="center">
                                <Segment className="center aligned" padded>
                                    <Label attached="top">{key}</Label>
                                    <Image
                                        src={`data:image/png;base64,${value}`}
                                        alt={key}
                                        style={{
                                            maxWidth: '100%',
                                            height: 'auto',
                                            display: 'block',
                                            margin: '0 auto',
                                        }}
                                    />
                                </Segment>
                            </Grid.Column>
                        ))}
                    </Grid.Row>
                    <Grid.Row columns={4}>
                        {Object.entries(imagens).slice(2).map(([key, value]) => (
                            <Grid.Column key={key} textAlign="center">
                                <Segment className="center aligned" padded>
                                    <Label attached="top">{key}</Label>
                                    <Image
                                        src={`data:image/png;base64,${value}`}
                                        alt={key}
                                        style={{
                                            maxWidth: '100%',
                                            height: 'auto',
                                            display: 'block',
                                            margin: '0 auto',
                                        }}
                                    />
                                </Segment>
                            </Grid.Column>
                        ))}
                    </Grid.Row>
                </>
            )}
        </>
    )}
</Grid>


        </Container>
    );
}

export default UploadImageArabic;
