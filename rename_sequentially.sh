for file in *."jpg"; do
    mv "$file" "$COUNTER.jpg"
    COUNTER=$[$COUNTER +1]
done