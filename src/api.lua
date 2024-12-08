
local socket = require "socket"

local data, msg_or_ip, port_or_nil

BalatrobotAPI = { }
BalatrobotAPI.socket = nil

BalatrobotAPI.waitingFor = nil
BalatrobotAPI.waitingForAction = true
BalatrobotAPI.lastAction = nil

function BalatrobotAPI.notifyapiclient()
    -- TODO Generate gamestate json object
    local _gamestateJsonString = ''
    if not BalatrobotAPI.waitingForAction then
        local _gamestate = {waitingFor = BalatrobotAPI.waitingFor, waitingForActin = BalatrobotAPI.waitingForAction, lastAction = BalatrobotAPI.lastAction}
        _gamestateJsonString = json.encode(_gamestate)
    else
        local _gamestate = Utils.getGamestate()
        _gamestate.waitingFor = BalatrobotAPI.waitingFor
        _gamestate.waitingForAction = BalatrobotAPI.waitingFor ~= nil and BalatrobotAPI.waitingForAction or false
        _gamestate.lastAction = BalatrobotAPI.lastAction
        _gamestateJsonString = json.encode(_gamestate)
    end

    if BalatrobotAPI.socket and port_or_nil ~= nil then
        BalatrobotAPI.socket:sendto(string.format("%s", _gamestateJsonString), msg_or_ip, port_or_nil)
    end
end

function BalatrobotAPI.respond(str)
    if BalatrobotAPI.socket and port_or_nil ~= nil then
        response = { }
        response.response = str
        str = json.encode(response)
        BalatrobotAPI.socket:sendto(string.format("%s\n", str), msg_or_ip, port_or_nil)
    end
end

function BalatrobotAPI.queueaction(action)
    Botlogger.reset()
    BalatrobotAPI.lastAction = action[1]
    local _params = Bot.ACTIONPARAMS[action[1]]
        List.pushleft(Botlogger['q_'.._params.func], { 0, action })
end

function BalatrobotAPI.update(dt)
    if not BalatrobotAPI.socket then
        sendDebugMessage('new socket')
        BalatrobotAPI.socket = socket.udp()
        BalatrobotAPI.socket:settimeout(0)
        local port = arg[1] or BALATRO_BOT_CONFIG.port
        BalatrobotAPI.socket:setsockname('0.0.0.0', tonumber(port))
    end

    data, msg_or_ip, port_or_nil = BalatrobotAPI.socket:receivefrom()
	if data then
        if data == 'MENU' then
            Middleware.conditionalactions = { }
            Middleware.queuedactions = List.new()
            BalatrobotAPI.waitingForAction = false
            BalatrobotAPI.waitingFor = 'start_run'
            BalatrobotAPI.lastAction = 'MENU'
            G.FUNCS.go_to_menu({ })
            BalatrobotAPI.respond("Menu command received")
        elseif data == 'HELLO\n' or data == 'HELLO' then
            BalatrobotAPI.notifyapiclient()
        else
            local _action = Utils.parseaction(data)
            local _err = Utils.validateAction(_action)

            if _err == Utils.ERROR.NUMPARAMS then
                BalatrobotAPI.respond("Error: Incorrect number of params for action " .. _action[1])
            elseif _err == Utils.ERROR.MSGFORMAT then
                BalatrobotAPI.respond("Error: Incorrect message format. Should be ACTION|arg1|arg2")
            elseif _err == Utils.ERROR.INVALIDACTION then
                BalatrobotAPI.respond("Error: Action invalid for action " .. _action[1])
            elseif _err == Utils.ERROR.WRONGACTION then
                BalatrobotAPI.respond("Error: Wrong action for current state " .. _action[1].. ' ' .. BalatrobotAPI.waitingFor)
            elseif BalatrobotAPI.waitingForAction then
                BalatrobotAPI.queueaction(_action)
                BalatrobotAPI.waitingForAction = false
                BalatrobotAPI.respond("Action queued"..data)
            else
                BalatrobotAPI.respond("Error: Not ready for action " .. _action[1])
            end
        end
	elseif msg_or_ip ~= 'timeout' then
		sendDebugMessage("Unknown network error: "..tostring(msg_or_ip))
    end

    -- No idea if this is necessary
    -- Without this being commented out, FPS capped out at ~80 for me
	-- socket.sleep(0.01)
end

function BalatrobotAPI.init()
    love.update = Hook.addcallback(love.update, BalatrobotAPI.update)

    -- Tell the game engine that every frame is 8/60 seconds long
    -- Speeds up the game execution
    -- Values higher than this seem to cause instability
    if BALATRO_BOT_CONFIG.dt then
        love.update = Hook.addbreakpoint(love.update, function(dt)
            return BALATRO_BOT_CONFIG.dt
        end)
    end

    -- Disable FPS cap
    if BALATRO_BOT_CONFIG.uncap_fps then
        G.FPS_CAP = 999999.0
    end

    -- Makes things move instantly instead of sliding
    if BALATRO_BOT_CONFIG.instant_move then
        function Moveable.move_xy(self, dt)
            -- Directly set the visible transform to the target transform
            self.VT.x = self.T.x
            self.VT.y = self.T.y
        end
    end

    -- Forcibly disable vsync
    if BALATRO_BOT_CONFIG.disable_vsync then
        love.window.setVSync(0)
    end

    -- Disable card scoring animation text
    if BALATRO_BOT_CONFIG.disable_card_eval_status_text then
        card_eval_status_text = function(card, eval_type, amt, percent, dir, extra) end
    end

    -- Disable chip count ticking up/down incrementally, instead set it directly
    if BALATRO_BOT_CONFIG.disable_chip_easing then
        local original_add_event = G.E_MANAGER.add_event
        G.E_MANAGER.add_event = function(event, queue, front)
            if event.trigger == 'ease' and event.ease.ref_value == 'chips' then
                G.GAME.chips = event.ease.end_val
            elseif event.trigger == 'ease' and event.ease.ref_value == 'chip_total' then
                G.GAME.current_round.current_hand.chip_total = event.ease.end_val
            else
                original_add_event(event, queue, front)
            end
        end
    end

    -- Disable the background animation
    G.SETTINGS.reduced_motion = 1

    -- G.FUNCS.wipe_on = function(message, no_card, timefac, alt_colour) end
    -- G.FUNCS.wipe_off = function() end
    G.FUNCS.delay = function(time, queue) end

    -- Only draw/present every Nth frame
    local original_draw = love.draw
    local draw_count = 0
    love.draw = function()
        draw_count = draw_count + 1
        if draw_count % BALATRO_BOT_CONFIG.frame_ratio == 0 then
            original_draw()
        end
    end

    local original_present = love.graphics.present
    love.graphics.present = function()
        if draw_count % BALATRO_BOT_CONFIG.frame_ratio == 0 then
            original_present()
        end
    end
    
    sendDebugMessage('init api')
    if Bot.SETTINGS.api == true then
        Middleware.c_play_hand = Hook.addbreakpoint(Middleware.c_play_hand, function()
            BalatrobotAPI.waitingFor = 'select_cards_from_hand'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_select_blind = Hook.addbreakpoint(Middleware.c_select_blind, function()
            BalatrobotAPI.waitingFor = 'skip_or_select_blind'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_choose_booster_cards = Hook.addbreakpoint(Middleware.c_choose_booster_cards, function()
            BalatrobotAPI.waitingFor = 'select_booster_action'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_shop = Hook.addbreakpoint(Middleware.c_shop, function()
            sendDebugMessage('SELECT SHOP ACTION')
            BalatrobotAPI.waitingFor = 'select_shop_action'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_rearrange_hand = Hook.addbreakpoint(Middleware.c_rearrange_hand, function()
            BalatrobotAPI.waitingFor = 'rearrange_hand'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_rearrange_consumables = Hook.addbreakpoint(Middleware.c_rearrange_consumables, function()
            BalatrobotAPI.waitingFor = 'rearrange_consumables'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_use_or_sell_consumables = Hook.addbreakpoint(Middleware.c_use_or_sell_consumables, function()
            BalatrobotAPI.waitingFor = 'use_or_sell_consumables'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_rearrange_jokers = Hook.addbreakpoint(Middleware.c_rearrange_jokers, function()
            BalatrobotAPI.waitingFor = 'rearrange_jokers'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_sell_jokers = Hook.addbreakpoint(Middleware.c_sell_jokers, function()
            BalatrobotAPI.waitingFor = 'sell_jokers'
            BalatrobotAPI.waitingForAction = true
        end)
        Middleware.c_start_run = Hook.addbreakpoint(Middleware.c_start_run, function()
            BalatrobotAPI.waitingFor = 'start_run'
            BalatrobotAPI.waitingForAction = true
        end)
    end
end

return BalatrobotAPI