#!/bin/bash

# Путь к корзине на внешнем диске
TRASH="/Volumes/FileHouse/.Trashes"

echo "Очистка корзины на диске FileHouse..."

# Сначала удаляем все обычные файлы
sudo find "$TRASH" -type f -exec rm -f {} \;

# Затем удаляем все скрытые системные файлы (например .DS_Store, ._*)
sudo find "$TRASH" -name ".*" -exec rm -f {} \;

# После этого удаляем пустые папки
sudo find "$TRASH" -type d -empty -exec rmdir {} \;

# И наконец пробуем снести всю корзину
sudo rm -rf "$TRASH"

echo "Готово. Корзина очищена."

