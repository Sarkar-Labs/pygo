package tensor

import "fmt"

func Fused0890CosAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Log2()
	return r, nil
}
func Fused0890CosAddLog2Name() string { return "cos_add_log2" }
