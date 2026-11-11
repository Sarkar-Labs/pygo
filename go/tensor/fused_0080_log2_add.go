package tensor

import "fmt"

func Fused0080Log2Add(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	return r, nil
}
func Fused0080Log2AddName() string { return "log2_add" }
