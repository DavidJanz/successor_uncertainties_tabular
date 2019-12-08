from numpy import mean


def q_train_iter(episode, ep_n, config, verbose):
    loss_list = []

    for _ in range(config.n_grad_steps):
        config.model.next_head()
        config.optimiser.zero_grad()
        batch = config.sample_fn(config.model.head)
        if batch is None:
            continue
        lq = config.loss(batch)
        lq.backward()

        config.optimiser.step()
        loss_list.append(lq.item())

    avg_q_loss = mean(loss_list)

    if ep_n % config.print_frequency == 0 and verbose:
        print(f"{ep_n}, {avg_q_loss:.3f}, {episode.reward:4.1f}")

    config.manager.tensorboard.add_scalar('training/q_loss', avg_q_loss, ep_n)
    config.manager.tensorboard.add_scalar('training/reward', episode.reward, ep_n)


def sf_train_iter(episode, ep_n, config, verbose):
    loss_list = []

    for _ in range(config.n_grad_steps):
        config.optimiser.zero_grad()

        batch = config.sample_fn()
        lr, lq, ltd = config.loss(batch)

        ltd.backward()
        config.model.r.zero_grad()
        lr.backward()
        lq.backward()

        config.optimiser.step()
        loss_list.append((lr.item(), lq.item(), ltd.item()))

    avg_r_loss, avg_q_loss, avg_td_loss = [mean(l) for l in zip(*loss_list)]

    if ep_n % config.print_frequency == 0 and verbose:
        print(f"{ep_n}, r_loss {avg_r_loss:.3f}, td_loss {avg_td_loss:.3f};    reward{episode.reward:4.1f}")

    config.manager.tensorboard.add_scalar('training/reward_loss', avg_r_loss, ep_n)
    config.manager.tensorboard.add_scalar('training/q_loss', avg_q_loss, ep_n)
    config.manager.tensorboard.add_scalar('training/td_loss', avg_td_loss, ep_n)
    config.manager.tensorboard.add_scalar('training/reward', episode.reward, ep_n)


def ube_train_iter(episode, ep_n, config, verbose):
    loss_list = []

    for _ in range(config.n_grad_steps):
        config.optimiser.zero_grad()
        batch = config.sample_fn()
        lq, lu = config.loss(batch)
        lq.backward()
        lu.backward()

        config.optimiser.step()
        loss_list.append((lq.item(), lu.item()))

    avg_q_loss, avg_ube_loss = [mean(l) for l in zip(*loss_list)]

    if ep_n % config.print_frequency == 0 and verbose:
        print(f"{ep_n}, {avg_q_loss:.3f}, {episode.reward:4.1f}")

    config.manager.tensorboard.add_scalar('training/q_loss', avg_q_loss, ep_n)
    config.manager.tensorboard.add_scalar('training/ube_loss', avg_ube_loss, ep_n)
    config.manager.tensorboard.add_scalar('training/reward', episode.reward, ep_n)
