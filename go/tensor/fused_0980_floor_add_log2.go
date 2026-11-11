package tensor

import "fmt"

func Fused0980FloorAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Log2()
	return r, nil
}
func Fused0980FloorAddLog2Name() string { return "floor_add_log2" }
