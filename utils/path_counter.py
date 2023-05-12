def count_paths(width, height, path_length, start):
    # Initialize a 3D list to store the number of paths to each cell
    num_paths = [[[0] * (path_length + 1) for _ in range(height)] for _ in range(width)]

    # There is exactly one path of length 0 to the starting cell
    num_paths[start[0]][start[1]][0] = 1

    # Calculate the number of paths to each cell for each path length
    for length in range(1, path_length + 1):
        for x in range(width):
            for y in range(height):
                # Check each of the four directions
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    # If the new cell is within the grid, add the number of paths to it
                    if 0 <= nx < width and 0 <= ny < height:
                        num_paths[x][y][length] += num_paths[nx][ny][length - 1]

    # Sum up the total number of paths of the desired length
    total_paths = 0
    for x in range(width):
        for y in range(height):
            total_paths += num_paths[x][y][path_length]

    return total_paths

def count_all_paths(width, height, path_length):
    # Initialize a 3D list to store the number of paths to each cell
    num_paths = [[[0] * (path_length + 1) for _ in range(height)] for _ in range(width)]

    # There is exactly one path of length 0 to each cell
    for x in range(width):
        for y in range(height):
            num_paths[x][y][0] = 1

    # Calculate the number of paths to each cell for each path length
    for length in range(1, path_length + 1):
        for x in range(width):
            for y in range(height):
                # Check each of the four directions
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    # If the new cell is within the grid, add the number of paths to it
                    if 0 <= nx < width and 0 <= ny < height:
                        num_paths[x][y][length] += num_paths[nx][ny][length - 1]

    # Sum up the total number of paths of the desired length
    total_paths = 0
    for x in range(width):
        for y in range(height):
            total_paths += num_paths[x][y][path_length]

    return total_paths


if __name__ == "__main__":
    total_paths = 0
    width = 12
    height = 12
    path_length = 59 # Here path_length = number of steps, so a path_length of 49 contains 50 vertices.
    total_paths = count_all_paths(width, height, path_length)
    print(f"The total number of path with length {path_length+1} in a {width}x{height} grid is {total_paths}.")
