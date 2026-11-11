package tensor

import "fmt"

func Fused0350TanhAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Log2()
	return r, nil
}
func Fused0350TanhAddLog2Name() string { return "tanh_add_log2" }
